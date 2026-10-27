#!/usr/bin/env python3
# 复现工具：按原配置逐条准备训练/验证数据，校验通过后清理对应原始 bag。
"""Sequential, resumable train/val preparation; run inside tmux.

Original preprocessing and SGBM parameters are unchanged. A failed stage stops
the queue and retains raw files. Reinvoke the same command to resume.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete-raw-after-check', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / 'output/local/trainval_prepare'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = yaml.safe_load((ROOT / 'config/dataset/tartandrive.yaml').read_text())
    local_path = ROOT / 'config/local/project.yaml'
    local = yaml.safe_load(local_path.read_text())
    def path(value):
        return Path(os.path.abspath(os.path.expandvars(str(value))))
    raw = path(local['paths']['raw_root'])
    processed = path(local['paths']['processed_root'])
    python = path(local['environments']['pf'])
    status_path = output / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'scenes': {}}
    status['delete_raw_after_check'] = args.delete_raw_after_check
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['OMP_NUM_THREADS'] = str(local.get('cpu_threads', 4))
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = str(python.parent.parent / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')

    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(status_path)

    def command(cmd, log):
        print('Running: ' + ' '.join(map(str, cmd)), flush=True)
        with log.open('a') as handle:
            subprocess.run(list(map(str, cmd)), env=env, cwd=ROOT,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    status['state'] = 'running'
    save()
    for split in ['train', 'val']:
        for scene in cfg[split + '_scenes']:
            state = status['scenes'].setdefault(scene, {'split': split, 'done': []})
            scene_logs = output / scene
            scene_logs.mkdir(exist_ok=True)
            try:
                for stage in ['download', 'extract', 'preprocess', 'depth', 'cuvslam', 'check-data', 'loader-check', 'cleanup']:
                    if stage in state['done']:
                        continue
                    status.update(current_scene=scene, current_stage=stage)
                    state['state'] = stage
                    save()
                    if shutil.disk_usage(ROOT).free < 80 * 2**30:
                        raise RuntimeError('Less than 80 GiB free; stopped before next stage')
                    if stage == 'loader-check':
                        # Exercise the original training loader, including real map crops.
                        code = """import sys, torch
from dataset.tartandrive import TartanDriveDataset
d=TartanDriveDataset(root=sys.argv[1],scene=sys.argv[2],geo_tiff_path=sys.argv[3],ground_image_resize=(512,512),aerial_image_resize=(768,768),augment=False,sample_poses=False)
assert len(d)>0
for i in sorted({0,len(d)//2,len(d)-1}):
 x=d[i]
 for k,v in x.items():
  if torch.is_tensor(v): assert torch.isfinite(v).all(), (i,k)
print('Original training loader passed:',len(d),'frames')
"""
                        command([python, '-c', code, processed, scene, path(local['paths']['map'])], scene_logs / 'loader-check.log')
                    elif stage == 'cleanup':
                        if args.delete_raw_after_check:
                            target = raw / 'bags' / scene
                            if target.is_symlink() or raw.is_symlink():
                                raise RuntimeError('Unexpected raw symlink; refusing cleanup')
                            if target.exists():
                                shutil.rmtree(target)
                            print(f'Validated; removed raw: {scene}', flush=True)
                    else:
                        command([python, ROOT / 'scripts/local/run.py', stage,
                                 '--config', local_path, '--scene', scene,
                                 '--output', scene_logs / stage], scene_logs / (stage + '.log'))
                    state['done'].append(stage)
                    save()
                state['state'] = 'complete'
                save()
            except Exception as exc:
                state.update(state='failed', error=str(exc))
                status['state'] = 'failed'
                save()
                raise
    status['state'] = 'complete'
    save()
    print('All configured train/val scenes prepared.', flush=True)


if __name__ == '__main__':
    main()
