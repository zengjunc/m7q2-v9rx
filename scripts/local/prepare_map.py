#!/usr/bin/env python3
# 复现工具：下载并核验原作者 TartanDrive2 地图，为 BEVLoc 生成同源 VRT。
"""Download the official TartanDrive2 GeoTIFF and build its BEVLoc VRT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio
from rasterio.shutil import copy as copy_raster

ROOT = Path(__file__).resolve().parents[2]
NAME = 'gascola_EPSG32617_scale03_082725.tiff'
URL = 'https://web.corral.tacc.utexas.edu/texasrobotics/web_UT-SARA-GQ/supplementary/TartanDrive2/' + NAME


def prepare(path):
    with rasterio.open(path) as source:
        if source.crs.to_epsg() != 32617 or not np.allclose(source.res, [.3, .3], atol=1e-6):
            raise ValueError(f'Expected EPSG:32617 and 0.3m pixels, got {source.crs}, {source.res}')
        source.read(window=rasterio.windows.Window(0, 0, min(64, source.width), min(64, source.height)))
        info = dict(source=URL, crs=str(source.crs), resolution=list(source.res),
                    width=source.width, height=source.height, bands=source.count,
                    bounds=list(source.bounds))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    info['sha256'] = digest.hexdigest()
    vrt = path.with_suffix('.vrt')
    copy_raster(path, vrt, driver='VRT')
    path.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, default=ROOT/'data/maps'/NAME)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download and not args.map.exists():
        args.map.parent.mkdir(parents=True, exist_ok=True)
        partial = args.map.with_suffix(args.map.suffix+'.part')
        subprocess.run(['curl', '-fL', '--retry', '3', '-C', '-', URL, '-o', str(partial)], check=True)
        # Validate before installing the completed download.
        with rasterio.open(partial) as source:
            source.read(window=rasterio.windows.Window(0, 0, 1, 1))
        partial.replace(args.map)
    print(json.dumps(prepare(args.map), indent=2))


if __name__ == '__main__':
    main()
