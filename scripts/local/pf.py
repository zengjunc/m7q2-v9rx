# 复现适配：加载公开权重并调用原作者 PF 循环；可选无效深度屏蔽或双目软聚合。
"""LOCAL ADDITION: headless upstream PF adapter with optional invalid-depth masking."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir', required=True, type=Path)
    p.add_argument('--map', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, help='Override the PF random seed for paired comparisons')
    variants = p.add_mutually_exclusive_group()
    variants.add_argument('--mask-invalid-depth', action='store_true', help='Exclude nonpositive/nonfinite depth from BEV geometry')
    variants.add_argument('--stereo-pooling', action='store_true', help='Soft-weight LR contradictions inside BEV pooling; includes valid-depth geometry')
    p.add_argument('--stereo-conflict-weight', type=float, default=.25, help='Fixed contradiction factor; 1 is the identity control')
    a = p.parse_args()
    if not 0 < a.stereo_conflict_weight <= 1:
        p.error('--stereo-conflict-weight must be in (0, 1]')
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    if a.seed is not None and not 0 <= a.seed < 2**32:
        p.error('--seed must be between 0 and 2**32-1')
    import hydra
    import timm
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    import run_pf as upstream
    from particle_filter import ParticleFilter
    from utils.misc import load_model, seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required')
    torch.set_num_threads(4)
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'config'), version_base='1.3'):
        cfg = hydra.compose(config_name='run_pf')
    cfg.sequence.root = str(a.scene_dir.resolve().parent)
    cfg.sequence.scenes = [a.scene_dir.name]
    cfg.sequence.geo_tiff_path = str(a.map.resolve())
    cfg.sequence.stereo_pooling = a.stereo_pooling
    cfg.sequence.stereo_conflict_weight = a.stereo_conflict_weight
    cfg.model.mask_invalid_depth = a.mask_invalid_depth
    cfg.ckpt_path = str(a.checkpoint.resolve())
    cfg.output_dir = str(a.output.resolve())
    if a.seed is not None:
        cfg.seed = a.seed
    seed_everything(cfg.seed)
    # The provided checkpoint contains both backbones: no secondary network download.
    create_model = timm.create_model
    def offline_backbone(*args, **kwargs):
        kwargs['pretrained'] = False
        return create_model(*args, **kwargs)
    timm.create_model = offline_backbone
    try:
        model = load_model(cfg.model, cfg.ckpt_path, device='cuda', strict_model=True, eval_mode=True)
    finally:
        timm.create_model = create_model
    pf = ParticleFilter(model, **cfg.run)
    loader = upstream.create_sequence_loader(cfg, pf, a.scene_dir.name)
    if a.limit:
        loader = DataLoader(Subset(loader.dataset, range(min(a.limit, len(loader.dataset)))),
                            batch_size=1, shuffle=False, num_workers=0)
    class HeadlessLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
    upstream.RerunLogger = HeadlessLogger
    a.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, a.output / 'config.yaml', resolve=True)
    # No copy of the prediction algorithm: execute the original implementation.
    upstream.run_pf(pf, loader, a.output, cfg)
    (a.output / 'run_options.json').write_text(json.dumps({
        'mask_invalid_depth': a.stereo_pooling or a.mask_invalid_depth,
        'stereo_pooling': a.stereo_pooling,
        'stereo_conflict_weight': a.stereo_conflict_weight if a.stereo_pooling else None,
        'seed': int(cfg.seed),
    }, indent=2) + '\n')
    print(f'Saved upstream PF output: {a.output / "pf_traj.csv"}')


if __name__ == '__main__':
    main()
