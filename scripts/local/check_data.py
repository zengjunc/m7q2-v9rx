# 复现工具：逐步检查已处理轨迹的文件、时间、相机配对、图像、深度与 VO；只输出诊断和建议，不修改输入。
"""Read-only checks for processed TartanDrive sequences; quality flags need review."""
import argparse
from collections import Counter
import csv
import filecmp
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (544, 1024)  # Current rectified camera calibration: height, width.
ADVICE = {
    'rgb_invalid': '重新提取图像；恢复前不使用该帧视觉观测。',
    'stereo_invalid': '重新提取并核实左右目尺寸，再生成深度和VO。',
    'depth_invalid': '核实毫米uint16格式并重新生成深度。',
    'depth_empty': '整帧没有有效深度；恢复前应跳过地图观测，不能按有效RGB-D使用。',
    'reference_invalid': '修复参考位姿和图像的时间对应；不可用该帧评估。',
    'vo_invalid': '核实VO文件及时间对应，不能将异常运动当作可靠输入。',
    'time_invalid': '核实时间戳及提取顺序，不对重复时刻直接插值。',
    'time_gap': '检查缺帧或预处理裁剪；按真实时间传播运动。',
    'sync_offset': '核实来源时间并重新配对相机，之后重算深度和VO。',
    'sync_unverified': '缺少可靠来源对应，先补齐同步检查依据。',
    'blur': '检查图像；可考虑降低或跳过视觉更新并保留运动传播。',
    'exposure': '检查过暗或过曝区域；可考虑降低视觉观测权重。',
    'depth_sparse': '检查双目匹配、同步和遮挡；不要直接填充大片空洞。',
    'vo_untracked': '跟踪器未返回位姿；保留上次位姿不代表车辆静止。',
    'vo_status_unknown': '补齐VO跟踪状态；仅凭重复位姿不能判断车辆静止或跟踪失败。',
    'motion_jump': '核实真实运动与VO，不仅凭大运动就删除帧。',
}


def table(path, columns, issues):
    try:
        frame = pd.read_csv(path)
        return frame[list(columns)].to_numpy(dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        issues.append(f'{path.name}: {exc}')
        return None


def source_pairs(extracted, stamps):
    """Reconstruct upstream pairing, then require byte-identical images per frame.

    Preserve raw header timestamps before upstream duplicate redistribution.
    Reconstructed associations alone are not proof of processed image provenance.
    """
    if extracted is None or not extracted.is_dir():
        return None, '未提供有效提取目录，无法核实原始相机配对。'
    sys.path.insert(0, str(ROOT / 'preprocessing'))
    frames = {}
    try:
        from preprocess_tartandrive import redistribute_timestamps
        for name in ['color', 'left', 'right']:
            t = np.atleast_1d(np.loadtxt(extracted / f'timestamp_multisense_{name}.txt'))
            paths = sorted((extracted / 'multisense' / name).glob('*.png'))
            if len(t) != len(paths) or not len(t) or not np.isfinite(t).all():
                raise ValueError(f'{name}: 源时间戳与图像不对应')
            f = pd.DataFrame({f't_{name}': t, f'raw_{name}': t,
                              f'path_{name}': [str(p) for p in paths]})
            redistribute_timestamps(f, ts_col=f't_{name}')
            frames[name] = f
        pairs = pd.merge_asof(frames['color'], frames['left'], left_on='t_color',
                              right_on='t_left', direction='nearest', tolerance=.2)
        pairs = pd.merge_asof(pairs, frames['right'], left_on='t_color',
                              right_on='t_right', direction='nearest', tolerance=.2)
        result = []
        times = pairs['t_color'].to_numpy()
        for t in stamps:
            if not np.isfinite(t):
                result.append(None)
                continue
            k = int(np.searchsorted(times, t))
            candidates = [j for j in [k-1, k] if 0 <= j < len(times)]
            j = min(candidates, key=lambda j: abs(times[j]-t))
            row = pairs.iloc[j]
            result.append(row if abs(times[j]-t) <= 1.1e-6 and
                          pd.notna(row['path_left']) and pd.notna(row['path_right']) else None)
        return result, '按原作者规则重建配对；逐帧核对三路图像字节后，才使用原始header时间差。'
    except (OSError, ValueError, KeyError, ImportError) as exc:
        return None, f'无法核实来源配对：{exc}'


def quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'median', 'p95', 'max'], map(float, np.quantile(a, [0, .5, .95, 1])))) if len(a) else None


def source_intervals(extracted, factor):
    """Inspect original exported camera logs separately from the processed subset."""
    if extracted is None or not extracted.is_dir():
        return {}
    result = {}
    for name in ['color', 'left', 'right']:
        try:
            t = np.atleast_1d(np.loadtxt(extracted / f'timestamp_multisense_{name}.txt'))
            if t.ndim != 1 or not np.isfinite(t).all():
                raise ValueError('源时间戳不是有效单列')
            dt = np.diff(t)
            positive = dt[dt > 0]
            median = float(np.median(positive)) if len(positive) else None
            result[name] = dict(frames=len(t), interval_s=quantiles(dt),
                                nonpositive_intervals=int((dt <= 0).sum()),
                                long_gap_source_indices=(np.flatnonzero(dt > factor*median)+1).tolist() if median else [])
        except (OSError, ValueError) as exc:
            result[name] = dict(error=str(exc))
    return result


def source_interval_check(timing):
    problems = []
    for name, item in timing.items():
        if 'error' in item:
            problems.append(f'{name}时间不可核实')
        elif item['nonpositive_intervals'] or item['long_gap_source_indices']:
            problems.append(f"{name}:非递增{item['nonpositive_intervals']}处，长间隔{len(item['long_gap_source_indices'])}处")
    return dict(status='review' if problems or len(timing) != 3 else 'pass',
                detail='；'.join(problems) if problems else '源相机时间无明显长间隔。' if len(timing) == 3 else '未提供原始相机时间。')


def write_outputs(output, report, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'frames.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['frame', 'status', 'flags'])
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    text = ['# 数据合格检查', '', f"场景：`{report['scene']}`", '',
            '只读诊断；review表示需要核查，不等于坏帧。没有删除帧、插值图像或修改PF。', '',
            '| 检查步骤 | 状态 | 说明 |', '|---|---|---|']
    for name, item in report['checks'].items():
        text.append(f"| {name} | {item['status']} | {item['detail']} |")
    text += ['', f"帧状态：{report['frame_counts']}", '',
             '## 问题与建议', '', '| 标记 | 数量 | 建议 |', '|---|---:|---|']
    for flag, count in report['flag_counts'].items():
        text.append(f'| {flag} | {count} | {ADVICE[flag]} |')
    text += ['', '阈值为可调整的初始启发式，不根据定位误差选帧；有效深度比例和清晰度不等于准确率。',
             '处理后时间间隔不能单独证明原始采集丢帧；来源配对不可核实时会明确标记。',
             '下一步自适应视觉更新应保留运动传播和完整评估时刻；VO也失败时不能假设运动可靠。']
    (output / 'REPORT.md').write_text('\n'.join(text)+'\n')
    plot_quality(output, report, rows)


def plot_quality(output, report, rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, key, label, threshold in [
        (axes[0], 'dt_s', 'Frame interval (s)', report['thresholds']['gap_s']),
        (axes[1], 'sharpness', 'Laplacian variance (width=512)', report['thresholds']['min_sharpness']),
        (axes[2], 'depth_valid_fraction', 'Valid depth fraction', report['thresholds']['min_depth_fraction']),
    ]:
        ax.plot([r['frame'] for r in rows], [r[key] for r in rows], linewidth=.8)
        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--', linewidth=.8)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    offsets = []
    for r in rows:
        left, right = r['left_color_offset_s'], r['right_color_offset_s']
        offsets.append(max(abs(left), abs(right), abs(left-right))*1000
                       if left is not None and right is not None else np.nan)
    axes[3].plot([r['frame'] for r in rows], offsets, linewidth=.8)
    axes[3].axhline(report['thresholds']['max_sync_ms'], color='red', linestyle='--', linewidth=.8)
    axes[3].set_ylabel('Camera time difference (ms)')
    axes[3].grid(alpha=.25)
    axes[-1].set_xlabel('Processed frame index')
    fig.tight_layout()
    fig.savefig(output / 'quality.png', dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extracted', type=Path)
    p.add_argument('--map', type=Path)
    p.add_argument('--min-sharpness', type=float, default=40)
    p.add_argument('--min-depth-fraction', type=float, default=.25)
    p.add_argument('--max-sync-ms', type=float, default=5)
    p.add_argument('--max-gap-factor', type=float, default=1.5)
    a = p.parse_args()
    if not a.scene_dir.is_dir() or a.output.resolve().is_relative_to(a.scene_dir.resolve()):
        p.error('Input must exist and output must be outside the scene directory')
    if not np.isfinite([a.min_depth_fraction, a.min_sharpness, a.max_sync_ms, a.max_gap_factor]).all() or not 0 <= a.min_depth_fraction <= 1 or a.min_sharpness < 0 or a.max_sync_ms < 0 or a.max_gap_factor <= 1:
        p.error('Invalid quality thresholds')
    cv2.setNumThreads(4)
    issues = []
    images = {name: sorted((a.scene_dir / name).glob('*.png')) for name in ['color', 'left', 'right', 'depth']}
    names = [x.name for x in images['color']]
    for name in ['left', 'right', 'depth']:
        if [x.name for x in images[name]] != names:
            issues.append(f'{name} 与 color 文件名/数量不一致')
    if not names:
        issues.append('没有color PNG')
    try:
        stamps = np.atleast_1d(np.loadtxt(a.scene_dir / 'timestamps.txt'))
        if stamps.ndim != 1:
            raise ValueError('时间戳必须为单列')
    except (OSError, ValueError) as exc:
        issues.append(f'timestamps.txt: {exc}')
        stamps = np.array([])
    if len(stamps) != len(names):
        issues.append('时间戳与图像数量不一致')
    ref = table(a.scene_dir / 'odom.csv', ['timestamp', 'x', 'y', 'angle'], issues)
    vo = table(a.scene_dir / 'pycuvslam.csv', ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], issues)
    for name, array in [('odom', ref), ('pycuvslam', vo)]:
        if array is not None and (len(array) != len(names) or not np.isfinite(array).all()):
            issues.append(f'{name} 数量或数值无效')
    tracked = {}
    try:
        entries = json.loads((a.scene_dir / 'pycuvslam.json').read_text())['statuses']
        tracked = {s['frame']: s['tracked'] for s in entries}
        if len(entries) != len(names) or set(tracked) != set(range(len(names))) or any(type(v) is not bool for v in tracked.values()):
            tracked = {}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    dt = np.diff(stamps)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive)) if len(positive) else None
    gap = median_dt*a.max_gap_factor if median_dt else None
    source_timing = source_intervals(a.extracted, a.max_gap_factor)
    pairs, source_note = source_pairs(a.extracted, stamps)
    rows = []
    for i, image in enumerate(images['color']):
        flags = []
        fatal = []
        t = float(stamps[i]) if i < len(stamps) and np.isfinite(stamps[i]) else None
        delta = float(stamps[i]-stamps[i-1]) if i > 0 and i < len(stamps) and np.isfinite(stamps[i-1:i+1]).all() else None
        if t is None or (i > 0 and (delta is None or delta <= 0)):
            fatal.append('time_invalid')
        elif gap is not None and delta is not None and delta > gap:
            flags.append('time_gap')
        for array, tag in [(ref, 'reference_invalid'), (vo, 'vo_invalid')]:
            if array is None or i >= len(array) or not np.isfinite(array[i]).all() or t is None or abs(array[i, 0]-t) > 1e-6:
                fatal.append(tag)
        if vo is not None and i < len(vo) and abs(np.linalg.norm(vo[i, 4:8])-1) > .001:
            fatal.append('vo_invalid')
        color = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(a.scene_dir / 'depth' / image.name), cv2.IMREAD_UNCHANGED)
        stereo = [cv2.imread(str(a.scene_dir / side / image.name), cv2.IMREAD_UNCHANGED) for side in ['left', 'right']]
        sharpness = exposure = fraction = None
        if color is None or color.shape != (*SHAPE, 3) or color.dtype != np.uint8:
            fatal.append('rgb_invalid')
        else:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (512, 272), interpolation=cv2.INTER_AREA)
            sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
            exposure = float(((gray <= 10) | (gray >= 245)).mean())
            if sharpness < a.min_sharpness:
                flags.append('blur')
            if exposure > .5:
                flags.append('exposure')
        if any(v is None or v.shape != SHAPE or v.dtype != np.uint8 for v in stereo):
            fatal.append('stereo_invalid')
        if depth is None or depth.shape != SHAPE or depth.dtype != np.uint16:
            fatal.append('depth_invalid')
        else:
            fraction = float(((depth >= 100) & (depth <= 65500)).mean())
            if np.any((depth > 0) & ((depth < 100) | (depth > 65500))):
                fatal.append('depth_invalid')
            if fraction < a.min_depth_fraction:
                flags.append('depth_sparse')
            if fraction == 0:
                fatal.append('depth_empty')
        left_offset = right_offset = None
        verified = False
        pair = pairs[i] if pairs is not None and i < len(pairs) else None
        if pair is not None:
            try:
                verified = all(filecmp.cmp(a.scene_dir / name / image.name, pair[f'path_{name}'], shallow=False)
                               for name in ['color', 'left', 'right'] if (a.scene_dir / name / image.name).is_file())
                verified &= all((a.scene_dir / name / image.name).is_file() for name in ['color', 'left', 'right'])
            except OSError:
                verified = False
        if verified:
            left_offset = float(pair['raw_left']-pair['raw_color'])
            right_offset = float(pair['raw_right']-pair['raw_color'])
            if max(abs(left_offset), abs(right_offset), abs(left_offset-right_offset))*1000 > a.max_sync_ms:
                flags.append('sync_offset')
        else:
            flags.append('sync_unverified')
        tracking = tracked.get(i)
        if tracking is False:
            flags.append('vo_untracked')
        elif tracking is None:
            flags.append('vo_status_unknown')
        speed = None
        if vo is not None and 0 < i < len(vo) and delta is not None and delta > 0 and np.isfinite(vo[i-1:i+1]).all():
            speed = float(np.linalg.norm(vo[i, 1:4]-vo[i-1, 1:4])/delta)
            if speed > 25:
                flags.append('motion_jump')
        flags = list(dict.fromkeys(fatal+flags))
        rows.append(dict(frame=i, filename=image.name, timestamp=t, dt_s=delta,
                         status='invalid' if fatal else 'review' if flags else 'ok', flags='|'.join(flags),
                         sharpness=sharpness, extreme_exposure_fraction=exposure,
                         depth_valid_fraction=fraction, source_images_verified=bool(verified),
                         left_color_offset_s=left_offset, right_color_offset_s=right_offset,
                         vo_tracked=tracking, vo_speed_m_s=speed,
                         suggested_action=' '.join(ADVICE[f] for f in flags)))
        if i % 200 == 0 or i == len(names)-1:
            print(f'Check {i+1}/{len(names)}', flush=True)
    counts = Counter(r['status'] for r in rows)
    flags = Counter(f for r in rows for f in r['flags'].split('|') if f)
    checks = {'文件与对应': dict(status='fail' if issues else 'pass', detail='；'.join(issues) or f'{len(names)}帧，四路PNG文件名与位姿数量对应。'),
              '原始消息时间': source_interval_check(source_timing),
              '时间轴': dict(status='fail' if flags['time_invalid'] else 'review' if flags['time_gap'] else 'pass', detail=f'间隔中位数{median_dt}秒；长间隔{flags["time_gap"]}处。'),
              '相机配对': dict(status='review' if flags['sync_offset'] or flags['sync_unverified'] else 'pass', detail=source_note),
              '图像质量': dict(status='fail' if flags['rgb_invalid'] or flags['stereo_invalid'] else 'review' if flags['blur'] or flags['exposure'] else 'pass', detail='固定512像素宽度计算清晰度；极暗/极亮像素比例阈值0.5。'),
              '深度': dict(status='fail' if flags['depth_invalid'] or flags['depth_empty'] else 'review' if flags['depth_sparse'] else 'pass', detail='检查uint16毫米、0无效及0.1至65.5米有效范围。'),
              'VO': dict(status='fail' if flags['vo_invalid'] else 'review' if flags['vo_untracked'] or flags['motion_jump'] or len(tracked) != len(names) else 'pass', detail=f'跟踪状态记录{len(tracked)}/{len(names)}；速度超过25m/s仅提示核查。')}
    if a.map:
        import rasterio
        try:
            with rasterio.open(a.map) as raster:
                b = raster.bounds
                coverage = float(((ref[:, 1] >= b.left) & (ref[:, 1] <= b.right) & (ref[:, 2] >= b.bottom) & (ref[:, 2] <= b.top)).mean()) if ref is not None and len(ref) else 0
                checks['地图覆盖'] = dict(status='pass' if coverage == 1 else 'fail', detail=f'参考位置覆盖比例{coverage:.4f}；CRS={raster.crs}；分辨率={raster.res}。只核查覆盖，不以定位误差选帧。')
        except (OSError, ValueError) as exc:
            checks['地图覆盖'] = dict(status='fail', detail=str(exc))
    report = dict(scene=a.scene_dir.name, frames=len(names), input_modified=False, checks=checks,
                  frame_counts=dict(counts), flag_counts=dict(flags),
                  thresholds=dict(min_sharpness=a.min_sharpness, min_depth_fraction=a.min_depth_fraction,
                                  max_sync_ms=a.max_sync_ms, gap_factor=a.max_gap_factor, gap_s=gap,
                                  max_extreme_fraction=.5, max_vo_speed_m_s=25),
                  interval_s=quantiles(dt), approximate_rate_hz=1/median_dt if median_dt else None,
                  source_intervals=source_timing,
                  sharpness=quantiles([r['sharpness'] for r in rows]),
                  depth_valid_fraction=quantiles([r['depth_valid_fraction'] for r in rows]),
                  source_pairing_note=source_note)
    write_outputs(a.output, report, rows)
    print(json.dumps({'frames':len(names), 'frame_counts':dict(counts), 'flag_counts':dict(flags),
                      'report':str(a.output / 'REPORT.md')}, ensure_ascii=False, indent=2))
    return 2 if issues or counts['invalid'] or any(c['status']=='fail' for c in checks.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
