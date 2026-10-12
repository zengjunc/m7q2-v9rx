#!/usr/bin/env python3
"""LOCAL ADDITION: shared project commands; upstream algorithms remain in src/."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'scripts/local'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['download','extract','preprocess','depth','prepare-bevloc','pf','train','cuvslam','superodom','bevloc'])
    p.add_argument('--config', type=Path, default=ROOT / 'config/local/project.yaml')
    p.add_argument('--scene')
    p.add_argument('--limit', type=int, help='PF / cuVSLAM frame limit, or BEVLoc batch limit')
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=Path, help='Override PF/BEVLoc inference weights for this run')
    p.add_argument('--dry-run', action='store_true')
    a, extra = p.parse_known_args()
    if a.checkpoint is not None and a.command not in ['pf', 'bevloc']:
        p.error('--checkpoint is only supported for pf/bevloc; train uses ckpt_path=...')
    if extra and a.command != 'train':
        p.error(f'Unknown arguments: {extra}')
    if a.limit is not None and (a.limit < 1 or a.command not in ['pf','cuvslam','bevloc']):
        p.error('--limit must be positive and is only supported for pf/cuvslam/bevloc')
    if not a.config.exists():
        p.error('Copy config/local/project.example.yaml to project.yaml and edit local paths')
    cfg = yaml.safe_load(a.config.read_text())
    def path(value):
        value = Path(os.path.expandvars(str(value))).expanduser()
        # Preserve venv interpreter symlinks: resolving them would bypass the venv.
        return Path(os.path.abspath(value if value.is_absolute() else ROOT / value))
    def loc(key): return path(cfg['paths'][key])
    scene = a.scene or cfg['scene']
    if Path(scene).name != scene or scene in ['.', '..']:
        p.error('Invalid scene name')
    data = loc('processed_root') / scene
    extracted = loc('extracted_root') / scene
    prepared = loc('bevloc_data_root') / scene
    output = path(a.output) if a.output else loc('output_root') / scene / a.command
    kind = 'bevloc' if a.command == 'bevloc' else 'cuvslam' if a.command == 'cuvslam' else 'pf'
    python = path(cfg['environments'][kind]) if '/' in cfg['environments'][kind] else cfg['environments'][kind]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), str(ROOT/'third_party/rosbagkit/src'), str(ROOT/'third_party/bevloc'), str(ROOT/'third_party/robot-eye'), env.get('PYTHONPATH','')])
    env['OMP_NUM_THREADS'] = str(cfg.get('cpu_threads', 4))
    if kind in ['pf', 'cuvslam']:
        library = path(cfg['environments']['pf']).parent.parent / 'lib'
        env['LD_LIBRARY_PATH'] = str(library) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if kind == 'bevloc':
        prefix = path(cfg['environments']['bevloc_native_prefix'])
        env['PROJ_DATA'] = str(prefix/'share/proj')
        env['LD_LIBRARY_PATH'] = str(prefix/'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    commands = []
    cmd = [str(python)]
    if a.command == 'download':
        commands = [cmd + [str(LOCAL/'download.py'),'--scene',scene,'--output',str(loc('raw_root'))]]
    elif a.command == 'extract':
        manifest = yaml.safe_load((ROOT/'third_party/tartandrive/assets/files.yaml').read_text())['bags']
        key = f'bags/{scene}/'
        if key not in manifest: p.error('Scene is not in the TartanDrive manifest')
        topics = yaml.safe_load((ROOT/'third_party/rosbagkit/config/extract/tartandrive2.yaml').read_text())['topics']
        extraction = dict(bagfile_root=str(loc('raw_root')), output_root=str(loc('extracted_root')),
                          timestamp_source='header', topics=topics,
                          scenes={scene:dict(bagfiles=[x for x in manifest[key]['files'] if x.endswith('.bag')])})
        config_path = loc('output_root')/'data_configs'/f'{scene}.yaml'
        if not a.dry_run:
            config_path.parent.mkdir(parents=True,exist_ok=True)
            config_path.write_text(yaml.safe_dump(extraction,sort_keys=False))
        commands = [cmd+[str(ROOT/'third_party/rosbagkit/scripts/extract_bagfile.py'),str(config_path)]]
    elif a.command == 'preprocess':
        # Upstream script processes all directories under --src: stage exactly one scene.
        stage = loc('output_root')/'data_configs'/f'{scene}_source'
        if not a.dry_run:
            if not extracted.is_dir(): p.error(f'Missing extracted scene: {extracted}')
            stage.mkdir(parents=True,exist_ok=True)
            link=stage/scene
            if not link.exists(): link.symlink_to(extracted,target_is_directory=True)
            if link.resolve()!=extracted: p.error(f'Unexpected staged source: {link}')
        commands = [cmd+[str(ROOT/'preprocessing/preprocess_tartandrive.py'),'--src',str(stage),'--dst',str(loc('processed_root'))]]
    elif a.command == 'depth':
        commands = [cmd+[str(LOCAL/'depth.py'),'--scene-dir',str(data)]]
    elif a.command == 'prepare-bevloc':
        commands = [cmd+[str(LOCAL/'prepare_bevloc.py'),'--processed',str(data),'--extracted',str(extracted),
                         '--bags',str(loc('raw_root')/'bags'/scene),'--output',str(prepared)]]
    elif a.command == 'pf':
        commands = [cmd+[str(LOCAL/'pf.py'),'--scene-dir',str(data),'--map',str(loc('map')),
                         '--checkpoint',str(path(a.checkpoint) if a.checkpoint else loc('pf_checkpoint')),'--output',str(output)]]
        if a.limit: commands[0] += ['--limit',str(a.limit)]
        commands += [cmd+[str(LOCAL/'evaluate.py'),'--method','pf','--scene-dir',str(data),'--output',str(output)]]
    elif a.command == 'train':
        # Invoke the original training pipeline; only select the TartanDrive dataset/config.
        commands = [cmd+[str(ROOT/'src/train.py'),'--config-name=train_tartandrive',
                         f'datasets.0.root={loc("processed_root")}',f'datasets.0.geo_tiff_path={loc("map")}',
                         f'output_dir={output}', 'wandb=False', *extra]]
    elif a.command == 'cuvslam':
        commands = [cmd+[str(LOCAL/'track_cuvslam.py'),'--scene-dir',str(data)]]
        if a.limit: commands[0] += ['--limit',str(a.limit)]
        eval_cmd = [str(path(cfg['environments']['pf'])),str(LOCAL/'evaluate.py'),'--method','cuvslam',
                    '--scene-dir',str(data),'--output',str(output)]
        if a.limit: eval_cmd += ['--limit',str(a.limit)]
        commands += [eval_cmd]
    elif a.command == 'superodom':
        commands = [cmd+[str(LOCAL/'evaluate.py'),'--method','superodom','--scene-dir',str(data),
                         '--extracted',str(extracted),'--output',str(output)]]
    else:
        settings = json.loads((ROOT/'config/local/bevloc.json').read_text())
        settings.update(map_file=str(loc('bevloc_map')),vrt_file=str(loc('bevloc_vrt')),
                        out_path=str(output),checkpoint_out_path=str(output),model_ckpt_path=str(path(a.checkpoint) if a.checkpoint else loc('bevloc_checkpoint')))
        settings['training'].update(train_split=str(loc('bevloc_data_root')),val_split=str(loc('bevloc_data_root')),test_split=str(loc('bevloc_data_root')))
        settings_path = output/'settings.json'
        if not a.dry_run:
            output.mkdir(parents=True,exist_ok=True)
            settings_path.write_text(json.dumps(settings,indent=2)+'\n')
        commands = [cmd+['-m','aeromatch.eval.eval_tartan_drive','--settings',str(settings_path),
                         '--checkpoint',str(path(a.checkpoint) if a.checkpoint else loc('bevloc_checkpoint')),'--trajectory',str(prepared),'--output-dir',str(output)]]
        if a.limit: commands[0] += ['--max-batches',str(a.limit)]
        # Evaluator only reads saved CSVs and can use the BEVLoc environment too.
        commands += [cmd+[str(LOCAL/'evaluate_bevloc.py'),'--run',str(output),
                          '--input-frames',str(sum(1 for _ in (prepared/'image_left_color/timestamps.txt').open()))]] if prepared.exists() else []
    for command in commands:
        print(shlex.join(command),flush=True)
        if not a.dry_run:
            subprocess.run(command,cwd=ROOT,env=env,check=True)


if __name__ == '__main__':
    main()
