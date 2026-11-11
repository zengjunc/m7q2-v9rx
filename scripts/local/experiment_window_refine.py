#!/usr/bin/env python3
# 复现新增：固定参数对照短窗口与异常触发，保存所有结果，不按测试真值自动选参数。
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from window_refine import analyze, evaluate, posterior_weights, refine


VARIANTS = {
    'identity': dict(lag=0),
    'window_all': dict(trigger='all'),
    'jump_only': dict(trigger='jump'),
    'valley_only': dict(trigger='valley'),
    'gated_lag3': dict(lag=3),
    'gated_lag5': {},
    'gated_lag10': dict(lag=10),
    'gated_weak': dict(strength=.25),
    'gated_strong': dict(strength=4.),
    'mode_centers': dict(mode_centers=True),
}
DEFAULT = dict(lag=5, strength=1., trigger='either', mode_centers=False, max_correction=.5)


def figures(folder, record, stats, corrected, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base, stamps = record['pf_pose'], record['timestamps']
    time = stamps-stamps[0]
    idx = np.searchsorted(reference[:, 0], stamps)
    origin = base[0, :2]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for x, label in [(reference[idx, 1:3], 'Reference'), (base[:, :2], 'PF'), (corrected[:, :2], 'Gated lag5')]:
        ax[0].plot(*(x-origin).T, label=label, lw=1)
    ax[0].set_aspect('equal', adjustable='datalim')
    ax[0].set(xlabel='East (m)', ylabel='North (m)')
    ax[0].legend()
    ax[1].plot(time, stats['innovation'], label='PF odometry residual')
    ax[1].plot(time, stats['support_ratio'], label='Mean support ratio', alpha=.7)
    ax[1].legend()
    ax[1].set(xlabel='Time (s)')
    ax[2].plot(time, np.linalg.norm(base[:, :2]-reference[idx, 1:3], axis=1), label='PF')
    ax[2].plot(time, np.linalg.norm(corrected[:, :2]-reference[idx, 1:3], axis=1), label='Gated lag5')
    ax[2].set(xlabel='Time (s)', ylabel='Position error (m)')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(folder/'overview.png', dpi=150)
    plt.close(fig)
    # Pick cases by diagnostics, not by improvement against reference.
    selected = [int(np.argmax(stats['innovation']))]
    if stats['valley'].any():
        selected.append(int(np.where(stats['valley'], stats['support_ratio'], np.inf).argmin()))
    weights = posterior_weights(record)
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 5*len(selected)), squeeze=False)
    for row, t in enumerate(selected):
        origin = base[t, :2]
        cloud = record['poses'][t, :, :2]-origin
        ax = axes[row, 0]
        ax.scatter(*cloud.T, s=10+300*weights[t]/weights[t].max(), c=weights[t], cmap='Greens', alpha=.6)
        ax.scatter(0, 0, marker='x', c='black', s=90, label='PF mean')
        ax.scatter(*(stats['centers'][t]-origin).T, marker='D', c='orange', label='Two-means centers')
        ax.scatter(*(reference[idx[t], 1:3]-origin), marker='*', c='red', s=100, label='Reference (evaluation only)')
        ax.set(title=f'Frame {t+1}, support={stats["support_ratio"][t]:.3f}', xlabel='East (m)', ylabel='North (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend()
        s, e = max(0, t-20), min(len(base), t+21)
        ax = axes[row, 1]
        for points, label in [(reference[idx[s:e], 1:3], 'Reference'), (base[s:e, :2], 'PF'), (corrected[s:e, :2], 'Gated lag5')]:
            ax.plot(*(points-origin).T, '.-', label=label, ms=2)
        ax.set(title=f'Local path, frame {t+1}', xlabel='East (m)', ylabel='North (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'cases.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {name: dict(DEFAULT, **overrides) for name, overrides in VARIANTS.items()}
    (args.output/'experiment_settings.json').write_text(json.dumps(configs, indent=2)+'\n')
    rows = []
    for archive in sorted(args.records.rglob('candidates.npz')):
        metadata = json.loads(archive.with_name('record_options.json').read_text())
        scene = Path(metadata['scene_dir']).name
        seed = metadata['seed']
        with np.load(archive) as z:
            record = dict(z)
        stats = analyze(record)
        # Generate every trajectory before loading reference for evaluation.
        trajectories = {name: refine(record, stats, **options) for name, options in configs.items()}
        reference = np.atleast_2d(np.loadtxt(Path(metadata['scene_dir'])/'odom.csv', delimiter=',', skiprows=1))
        baseline = evaluate(record['pf_pose'], record, reference)
        folder = args.output/f'{scene}_seed{seed}'
        folder.mkdir(exist_ok=True)
        np.savetxt(folder/'diagnostics.csv', np.column_stack([record['timestamps'], stats['spread'],
            stats['support_ratio'], stats['innovation'], stats['valley'], stats['jump']]), delimiter=',',
            header='timestamp,spread_m,support_ratio,innovation_m,valley,jump', comments='')
        for name, (trajectory, gate) in trajectories.items():
            change = np.linalg.norm(trajectory[:, :2]-record['pf_pose'][:, :2], axis=1)
            metrics = evaluate(trajectory, record, reference)
            row = dict(scene=scene, seed=seed, variant=name,
                       valley_frames=int(stats['valley'].sum()), jump_frames=int(stats['jump'].sum()),
                       changed_frames=int(np.sum(change > 1e-8)), max_correction_m=float(change.max()),
                       **metrics)
            row.update({f'pf_{key}': value for key, value in baseline.items()})
            rows.append(row)
            np.savetxt(folder/f'{name}.csv', np.column_stack([record['timestamps'], trajectory]),
                       delimiter=',', header='timestamp,x,y,angle', comments='', fmt='%.9f')
            (folder/f'{name}.json').write_text(json.dumps(dict(options=configs[name], **row), indent=2)+'\n')
        figures(folder, record, stats, trajectories['gated_lag5'][0], reference)
        print(f'Finished {scene} seed={seed}: valley={stats["valley"].sum()}, jump={stats["jump"].sum()}', flush=True)
        with (args.output/'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'Complete: {len(rows)} paired experiments -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
