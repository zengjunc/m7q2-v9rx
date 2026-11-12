#!/usr/bin/env python3
# 复现新增：官方地图上的四组独立完整轨迹对照，按场景保存轨迹图和指标。
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
from threading import Lock

import numpy as np
from window_refine import evaluate

ROOT = Path(__file__).resolve().parents[2]
SCENES = [
    '2023-11-02-16-00-43_down_meadows',
    'warehouse_to_fence_2023-09-12-10-19-17',
    'turnpike_2023-09-12-12-39-19',
    'warehouse_gupta_skydio_2023-09-13-16-45-20',
    'turnpike_2023-09-13-18-09-45',
    'figure_8_2023-09-14-12-48-02',
]
METHODS = ['base', 'valid_depth', 'stereo', 'window']
PLOT_LOCK = Lock()


def command(args, folder, env):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.log').open('a') as log:
        log.write(' '.join(args)+'\n'); log.flush()
        subprocess.run([str(x) for x in args], cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def process(scene, gpu, args):
    import yaml
    cfg = yaml.safe_load((ROOT/'config/local/project.yaml' if (ROOT/'config/local/project.yaml').exists()
                          else ROOT/'config/local/project.example.yaml').read_text())
    scene_dir = ROOT/Path(cfg['paths']['processed_root'])/scene
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, MPLCONFIGDIR='/tmp/bev-compare-mpl', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(ROOT/'.env-unified/lib')+':'+env.get('LD_LIBRARY_PATH', '')
    folder = args.output/scene
    py = str(ROOT/'.env-unified/bin/python')
    print(f'START {scene} GPU={gpu}', flush=True)
    record = folder/'base'
    if not (record/'candidates.npz').exists():
        command([py, 'scripts/local/record_pf_candidates.py', '--scene', scene, '--seed', str(args.seed),
                 '--gpu', gpu, '--output', str(record)], record, env)
    print(f'DONE {scene} base', flush=True)
    for method, flags in [('valid_depth', ['--mask-invalid-depth']), ('stereo', ['--stereo-pooling'])]:
        out = folder/method
        if not (out/'metrics.json').exists():
            command([py, 'scripts/local/run.py', 'pf', '--scene', scene, '--seed', str(args.seed),
                     '--output', str(out), *flags], out, env)
        print(f'DONE {scene} {method}', flush=True)
    out = folder/'window'
    if not (out/'metrics.json').exists():
        command([py, 'scripts/local/window_refine.py', '--record', str(record/'candidates.npz'),
                 '--lag', '5', '--output', str(out)], out, env)
    # Compare saved trajectories only; reference is not used by any method above.
    with np.load(record/'candidates.npz') as z:
        data = dict(z)
    reference = np.loadtxt(scene_dir/'odom.csv', delimiter=',', skiprows=1)
    trajectories = [np.loadtxt(folder/m/'pf_traj.csv', delimiter=',', skiprows=1) for m in METHODS[:3]]
    trajectories.append(np.loadtxt(out/'refined_traj.csv', delimiter=',', skiprows=1))
    rows = []
    for method, trajectory in zip(METHODS, trajectories):
        np.testing.assert_allclose(trajectory[:, 0], data['timestamps'], rtol=0, atol=1e-6)
        comparison_record = dict(data, pf_pose=trajectory[:, 1:4])
        rows.append(dict(scene=scene, method=method, seed=args.seed,
                         **evaluate(trajectory[:, 1:4], comparison_record, reference)))
    (folder/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    with PLOT_LOCK:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ref = reference[np.searchsorted(reference[:, 0], data['timestamps'])]
        origin = ref[0, 1:3]
        ax[0].plot(*(ref[:, 1:3]-origin).T, c='black', label='Reference')
        for method, trajectory in zip(METHODS, trajectories):
            ax[0].plot(*(trajectory[:, 1:3]-origin).T, label=method, lw=1)
            ax[1].plot(trajectory[:, 0]-trajectory[0, 0],
                       np.linalg.norm(trajectory[:, 1:3]-ref[:, 1:3], axis=1), label=method)
        ax[0].set_aspect('equal', adjustable='datalim');ax[0].set(xlabel='East (m)', ylabel='North (m)')
        ax[1].set(xlabel='Time (s)', ylabel='Position error (m)')
        for axis in ax:axis.legend();axis.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/'comparison.png', dpi=160);plt.close(fig)
    print(f'COMPLETE {scene}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'output/local/official_map_comparison')
    p.add_argument('--gpus', nargs='+', default=['0'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scenes', nargs='+', default=SCENES)
    args = p.parse_args();args.output = args.output.resolve();args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'experiment.json').write_text(json.dumps(dict(scenes=args.scenes,seed=args.seed,
        methods=METHODS, map='gascola_EPSG32617_scale03_082725.tiff',
        window=dict(lag=5, strength=1., max_correction=.5, trigger='either'),
        note='four independent arms: base, base+mask, base+stereo (includes mask), base+window (no mask/stereo); window adds approximately 0.5s latency'), indent=2)+'\n')
    def worker(pair):
        i,gpu = pair
        for scene in args.scenes[i::len(args.gpus)]:
            try:process(scene, gpu, args)
            except Exception as error:
                (args.output/f'{scene}.failed.txt').write_text(str(error)+'\n')
                print(f'FAILED {scene}: {error}', flush=True)
    with ThreadPoolExecutor(len(args.gpus)) as pool:list(pool.map(worker,enumerate(args.gpus)))
    rows = [r for scene in args.scenes if (args.output/scene/'comparison.json').exists()
            for r in json.loads((args.output/scene/'comparison.json').read_text())]
    if rows:
        with (args.output/'summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 官方地图四组独立对照', '', '六条完整测试轨迹。四组各自以原版base为基础；仅window组等待后续5帧。', '',
           '| 轨迹 | base RMSE/m | base+valid_depth | base+stereo | base+window |', '| --- | ---: | ---: | ---: | ---: |']
    for scene in args.scenes:
        subset=[r for r in rows if r['scene']==scene]
        values=[f"{next(r['RMSE_m'] for r in subset if r['method']==m):.6f}" if subset else 'failed' for m in METHODS]
        lines.append('| '+scene+' | '+' | '.join(values)+' |')
    lines += ['',f'完成 {len(rows)//4}/{len(args.scenes)} 条；失败详见 *.failed.txt。',
              '每条轨迹的 comparison.png 为轨迹与误差图，summary.csv 含位置与运动一致性指标。']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('BATCH FINISHED',flush=True)


if __name__ == '__main__':main()
