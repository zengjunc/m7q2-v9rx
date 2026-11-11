#!/usr/bin/env python3
# 复现新增：只记录原 PF 候选和权重，供短窗口修正实验复用，不改变原 PF。
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def motion_sigma(action, buffer, action_gain, drift_gain):
    """Mirror the original motion noise scale for scoring, without drawing RNG."""
    sigma = np.asarray(action_gain) * np.abs(action)
    if len(buffer) >= 2:
        sigma *= 1 + np.asarray(drift_gain) * (
            np.std(buffer, axis=0) / (np.mean(np.abs(buffer), axis=0) + 1e-6))
    return sigma


def make_recording_filter(base_class):
    """Subclass only records; all predict/update/resample operations stay upstream."""
    class RecordingFilter(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = {k: [] for k in ('poses', 'log_likelihood', 'actions',
                                         'motion_sigma', 'resampled', 'weights')}

        def initialize(self, init_xyr, init_noise_sigma=np.zeros(3)):
            super().initialize(init_xyr, init_noise_sigma)
            self.initial_pose = np.asarray(init_xyr).copy()
            self.initial_sigma = np.asarray(init_noise_sigma).copy()

        def predict(self, action, sigma=None):
            super().predict(action, sigma=sigma)
            self.trace['actions'].append(np.asarray(action).copy())
            mm = self.motion_model
            self.trace['motion_sigma'].append(motion_sigma(
                action, mm.action_buffer, mm.action_gain, mm.drift_gain))

        def update(self, ground_image, ground_depth, info, results=None):
            results = {} if results is None else results
            super().update(ground_image, ground_depth, info, results)
            self.trace['poses'].append(np.asarray(
                [[p.x, p.y, p.theta] for p in self.particles], dtype=np.float64))
            self.trace['weights'].append(np.array([p.weight for p in self.particles]))
            # Raw observation log likelihood, NOT normalized PF posterior weights.
            self.trace['log_likelihood'].append(
                np.asarray(results['log_likelihood'], dtype=np.float64).copy())

        def resample(self):
            sampled = super().resample()
            self.trace['resampled'].append(sampled)
            return sampled

    return RecordingFilter


def record_pf(args):
    import yaml
    config_path = args.config or ROOT / 'config/local/project.yaml'
    if not config_path.exists(): config_path = ROOT / 'config/local/project.example.yaml'
    config = yaml.safe_load(config_path.read_text())
    def location(name):
        value = Path(os.path.expandvars(str(config['paths'][name]))).expanduser()
        return value if value.is_absolute() else ROOT / value
    scene = args.scene or config['scene']
    scene_dir = location('processed_root') / scene
    # Keep setup isolated to this process; never change the upstream files.
    prefix_value = Path(config['environments']['pf']).parent.parent
    prefix = prefix_value if prefix_value.is_absolute() else ROOT / prefix_value
    os.environ.setdefault('PROJ_DATA', str(prefix / 'share/proj'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, str(ROOT / 'src'))
    import hydra
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    import run_pf as upstream
    from particle_filter import ParticleFilter
    from utils.misc import load_model, seed_everything
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to record real model observations; refinement works on CPU')
    torch.set_num_threads(4)
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'config'), version_base='1.3'):
        cfg = hydra.compose(config_name='run_pf')
    cfg.sequence.root = str(scene_dir.parent)
    cfg.sequence.scenes = [scene]
    cfg.sequence.geo_tiff_path = str(location('map'))
    cfg.sequence.stereo_pooling = args.stereo_pooling
    cfg.model.mask_invalid_depth = args.mask_invalid_depth
    OmegaConf.update(cfg, 'model.pretrained_backbones', False, force_add=True)
    cfg.ckpt_path = str(location('pf_checkpoint'))
    cfg.seed = args.seed
    seed_everything(cfg.seed)
    model = load_model(cfg.model, cfg.ckpt_path, device='cuda', strict_model=True, eval_mode=True)
    pf = make_recording_filter(ParticleFilter)(model, **cfg.run)
    loader = upstream.create_sequence_loader(cfg, pf, scene)
    if args.limit:
        loader = DataLoader(Subset(loader.dataset, range(min(args.limit, len(loader.dataset)))),
                            batch_size=1, shuffle=False, num_workers=0)
    class HeadlessLogger:
        def __init__(self, *a, **kw): pass
        def log(self, *a, **kw): pass
    upstream.RerunLogger = HeadlessLogger
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / 'config.yaml', resolve=True)
    metadata = dict(scene_dir=str(scene_dir.resolve()), seed=args.seed,
                    mask_invalid_depth=args.mask_invalid_depth or args.stereo_pooling,
                    stereo_pooling=args.stereo_pooling, gpu=args.gpu,
                    checkpoint=cfg.ckpt_path, limit=args.limit,
                    candidates='PF after predict/update, before resampling',
                    raw_observation='original alpha * score / likelihood_temp',
                    reference_usage='initial pose only during recording; subsequent reference poses only in upstream diagnostics')
    (args.output / 'record_options.json').write_text(json.dumps(metadata, indent=2) + '\n')
    upstream.run_pf(pf, loader, args.output, cfg)
    baseline = np.atleast_2d(np.loadtxt(args.output / 'pf_traj.csv', delimiter=',', skiprows=1))
    data = {key: np.asarray(value) for key, value in pf.trace.items()}
    data.update(initial_pose=pf.initial_pose, initial_sigma=pf.initial_sigma,
                resample_noise_sigma=np.asarray(cfg.run.resample_noise_sigma),
                timestamps=baseline[:, 0], pf_pose=baseline[:, 1:4])
    np.savez_compressed(args.output / 'candidates.npz', **data)
    print(f'Saved {len(baseline)} candidate layers: {args.output / "candidates.npz"}', flush=True)



def main():
    parser = argparse.ArgumentParser(description='Record original PF without changing its output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--scene')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', default='0')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--mask-invalid-depth', action='store_true')
    variants.add_argument('--stereo-pooling', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    record_pf(args)


if __name__ == '__main__':
    main()
