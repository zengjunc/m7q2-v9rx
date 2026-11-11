#!/usr/bin/env python3
# 复现新增：检测 PF 粒子分布与运动异常，有限延迟修正位置；不改变原 PF 和航向。
"""Fixed-lag position refinement using PF support and relative odometry, without GT."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


def posterior_weights(record):
    """Old caches have likelihoods and resampling flags; recover PF weights exactly."""
    if 'weights' in record:
        return record['weights']
    likelihood = record['log_likelihood']
    weights = np.empty_like(likelihood, dtype=float)
    previous = np.full(likelihood.shape[1], 1 / likelihood.shape[1])
    for t, observation in enumerate(likelihood):
        if t and record['resampled'][t-1]:
            previous = np.full_like(previous, 1 / len(previous))
        score = np.log(previous + 1e-30) + observation
        row = np.exp(score - score.max())
        weights[t] = previous = row / row.sum()
    return weights


def odometry_steps(record):
    """Body-frame translation -> map translation using previous PF heading."""
    yaw = np.r_[record['initial_pose'][2], record['pf_pose'][:-1, 2]]
    action = record['actions']
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c*action[:, 0]-s*action[:, 1],
                            s*action[:, 0]+c*action[:, 1]))


def analyze(record):
    """Weighted two-means + density valley; a broad cloud alone is not a trigger."""
    weights = posterior_weights(record)
    frames = len(weights)
    centers, masses = np.empty((frames, 2, 2)), np.empty((frames, 2))
    spread, separation, contrast = np.zeros((3, frames))
    for t, (pose, weight) in enumerate(zip(record['poses'], weights)):
        xy = pose[:, :2]
        mean = weight @ xy
        delta = xy - mean
        values, vectors = np.linalg.eigh((delta.T * weight) @ delta)
        spread[t] = np.sqrt(max(values.sum(), 0))
        center = mean + np.array([-1., 1.])[:, None] * np.sqrt(max(values[-1], 0)) * vectors[:, -1]
        for _ in range(8):
            label = ((xy[:, None] - center)**2).sum(-1).argmin(1)
            mass = np.bincount(label, weights=weight, minlength=2)
            if mass.min() < 1e-8:
                center[:] = mean
                mass = np.array([1., 0.])
                break
            center = np.array([(weight * (label == j)) @ xy / mass[j] for j in (0, 1)])
        centers[t], masses[t] = center, mass
        distance = np.linalg.norm(center[1]-center[0])
        axis = (center[1]-center[0]) / max(distance, 1e-8)
        projected = (xy-mean) @ axis
        within = sum(np.sum(weight*(label == j)*(projected-(center[j]-mean) @ axis)**2)
                     for j in (0, 1))
        bandwidth = max(.3, .5*np.sqrt(within))
        queries = (np.vstack([mean, center])-mean) @ axis
        density = np.exp(-.5*((projected[:, None]-queries)/bandwidth)**2).T @ weight
        contrast[t] = density[0] / max(density[1:].max(), 1e-12)
        separation[t] = distance
    base = record['pf_pose'][:, :2]
    steps = odometry_steps(record)
    previous = np.vstack([record['initial_pose'][:2], base[:-1]])
    innovation = np.linalg.norm(base-previous-steps, axis=1)
    # 0.25 m per nominal 0.1 s, adjusted for the actual sampling interval.
    timestamps = record['timestamps']
    dt = np.r_[np.median(np.diff(timestamps)) if frames > 1 else .1, np.diff(timestamps)]
    jump = innovation > .25*np.maximum(dt/.1, .5)
    jump[0] = False  # initialization is not a temporal jump
    valley = (masses.min(axis=1) >= .15) & (separation > 1.) & (contrast < .5)
    return dict(centers=centers, masses=masses, spread=spread, separation=separation,
                support_ratio=contrast, innovation=innovation, jump=jump, valley=valley,
                steps=steps)


def mode_targets(base, stats, start, stop, strength):
    """Tiny local DP on two supported cluster means; normal frames keep PF mean.

    Cluster masses are a heuristic support cost, NOT independent likelihoods.
    No hard motion gate and no unbounded history accumulation.
    """
    nodes, parents = [], []
    score = np.zeros(1)
    previous = base[start:start+1]
    for t in range(start, stop):
        poses = stats['centers'][t] if stats['valley'][t] else base[t:t+1]
        mass = stats['masses'][t] if stats['valley'][t] else np.ones(1)
        if t == start:
            cost = -np.log(mass + 1e-12)
            parent = np.zeros(len(poses), dtype=int)
        else:
            residual = poses[None]-previous[:, None]-stats['steps'][t]
            total = score[:, None] + strength*np.sum(residual**2, axis=-1)/.25**2
            parent = total.argmin(axis=0)
            cost = total[parent, np.arange(len(poses))]-np.log(mass + 1e-12)
        nodes.append(poses)
        parents.append(parent)
        previous, score = poses, cost
    slot = int(score.argmin())
    target = np.empty((stop-start, 2))
    for i in range(len(nodes)-1, -1, -1):
        target[i] = nodes[i][slot]
        slot = parents[i][slot]
    return target


def smooth_window(observations, steps, reliability, strength):
    """Minimize sum q|x-m|² + strength*sum|x[j]-x[j-1]-odo[j]|²."""
    n = len(observations)
    # Solve in local coordinates to avoid cancellation of large UTM coordinates.
    origin = observations[0]
    rhs = reliability[:, None] * (observations-origin)
    band = np.zeros((3, n))
    band[1] = reliability
    if n > 1:
        band[1, :-1] += strength
        band[1, 1:] += strength
        band[0, 1:] = band[2, :-1] = -strength
        rhs[:-1] -= strength*steps[1:]
        rhs[1:] += strength*steps[1:]
    return solve_banded((1, 1), band, rhs) + origin


def refine(record, stats=None, lag=5, strength=1., trigger='either',
           mode_centers=False, max_correction=.5):
    """Emit frame t after seeing at most t+lag; committed frames never change."""
    if lag < 0 or strength <= 0 or max_correction <= 0:
        raise ValueError('lag >= 0, strength > 0 and max_correction > 0 required')
    stats = analyze(record) if stats is None else stats
    base = np.asarray(record['pf_pose'], dtype=float)
    result = base.copy()
    events = dict(jump=stats['jump'], valley=stats['valley'],
                  either=stats['jump'] | stats['valley'],
                  all=np.ones(len(base), dtype=bool))[trigger]
    gate = np.zeros(len(base))
    for t in range(len(base)):
        start, stop = max(0, t-lag), min(len(base), t+lag+1)
        indices = np.arange(start, stop)
        gate[t] = np.max(events[start:stop]*(1-np.abs(indices-t)/(lag+1)))
        if gate[t] == 0 or lag == 0:
            continue
        observed = (mode_targets(base[:, :2], stats, start, stop, strength)
                    if mode_centers else base[start:stop, :2])
        reliability = np.ones(stop-start)
        reliability[stats['jump'][start:stop] | stats['valley'][start:stop]] = .25
        smoothed = smooth_window(observed, stats['steps'][start:stop], reliability, strength)
        delta = gate[t]*(smoothed[t-start]-base[t, :2])
        delta *= min(1., max_correction/max(np.linalg.norm(delta), 1e-12))
        result[t, :2] += delta
    return result, gate


def motion_metrics(poses, record):
    """Odometry agreement is a consistency proxy, not ground-truth accuracy."""
    dt = np.diff(record['timestamps'])
    residual = np.diff(poses[:, :2], axis=0)-odometry_steps(record)[1:]
    error = np.linalg.norm(residual, axis=1)
    # Turning/motion are subtracted, so ordinary vehicle acceleration is not penalized.
    velocity_residual = residual/dt[:, None]
    return dict(odo_residual_rms_m=float(np.sqrt(np.mean(error**2))),
                odo_residual_p95_m=float(np.percentile(error, 95)),
                odo_residual_max_m=float(error.max()),
                odo_residual_over_025m=int(np.sum(error > .25)),
                residual_velocity_change_rms_mps=float(np.sqrt(np.mean(
                    np.sum(np.diff(velocity_residual, axis=0)**2, axis=1)))))


def evaluate(poses, record, reference):
    # Called after refinement, never passed into analyze/refine.
    idx = np.searchsorted(reference[:, 0], record['timestamps'])
    if (idx >= len(reference)).any() or not np.allclose(reference[idx, 0], record['timestamps'], rtol=0, atol=1e-6):
        raise ValueError('Reference timestamps do not match')
    error = np.linalg.norm(poses[:, :2]-reference[idx, 1:3], axis=1)
    reference_step_error = np.linalg.norm(np.diff(poses[:, :2], axis=0)-
                                         np.diff(reference[idx, 1:3], axis=0), axis=1)
    return dict(frames=len(error), RMSE_m=float(np.sqrt(np.mean(error**2))),
                mean_m=float(error.mean()), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)),
                reference_step_rms_m=float(np.sqrt(np.mean(reference_step_error**2))),
                reference_step_p95_m=float(np.percentile(reference_step_error, 95)),
                **motion_metrics(poses, record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lag', type=int, default=5, help='future frames, usually 0.5 seconds')
    p.add_argument('--strength', type=float, default=1.)
    p.add_argument('--trigger', choices=['jump', 'valley', 'either', 'all'], default='either')
    p.add_argument('--mode-centers', action='store_true', help='experimental local mode selection')
    p.add_argument('--max-correction', type=float, default=.5, help='position correction cap in metres')
    args = p.parse_args()
    with np.load(args.record) as z:
        record = dict(z)
    stats = analyze(record)
    options = {k: getattr(args, k) for k in ['lag', 'strength', 'trigger', 'mode_centers', 'max_correction']}
    poses, gate = refine(record, stats, **options)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output/'refined_traj.csv', np.column_stack([record['timestamps'], poses]),
               delimiter=',', header='timestamp,x,y,angle', comments='', fmt='%.9f')
    np.savetxt(args.output/'diagnostics.csv', np.column_stack([
        record['timestamps'], stats['spread'], stats['support_ratio'], stats['innovation'],
        stats['valley'], stats['jump'], gate, np.linalg.norm(poses[:, :2]-record['pf_pose'][:, :2], axis=1)]),
        delimiter=',', header='timestamp,spread_m,support_ratio,innovation_m,valley,jump,gate,correction_m', comments='')
    metadata = json.loads(args.record.with_name('record_options.json').read_text())
    reference = np.atleast_2d(np.loadtxt(Path(metadata['scene_dir'])/'odom.csv', delimiter=',', skiprows=1))
    report = dict(options=options, record=str(args.record.resolve()),
                  pf=evaluate(record['pf_pose'], record, reference), refined=evaluate(poses, record, reference),
                  valley_frames=int(stats['valley'].sum()), jump_frames=int(stats['jump'].sum()),
                  changed_frames=int(np.sum(np.linalg.norm(poses[:, :2]-record['pf_pose'][:, :2], axis=1)>1e-8)),
                  latency='at most lag future frames; shortened lookahead at end of sequence',
                  scope='position-only output refinement; no feedback to PF, no GT in refinement')
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
