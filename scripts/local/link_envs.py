"""LOCAL ADDITION: bind existing environment prefixes to portable .envs/ names."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['pf', 'cuvslam', 'bevloc', 'bevloc-native']:
        p.add_argument('--' + name, type=Path, help='Existing environment prefix, not its bin/python')
    a = p.parse_args()
    bindings = vars(a)
    if not any(bindings.values()):
        p.error('Provide at least one environment prefix, e.g. --pf "$CONDA_PREFIX"')
    # Validate every requested binding before writing any links.
    requests = []
    for name, value in bindings.items():
        if value is None:
            continue
        target = value.expanduser().resolve()
        name = name.replace('_', '-')
        required = target / ('share/proj' if name == 'bevloc-native' else 'bin/python')
        if not required.exists():
            p.error(f'Missing {required}; pass an installed environment prefix')
        link = ROOT / '.envs' / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target:
                p.error(f'{link} already exists with another target; inspect it before changing')
        requests.append((link, target))
    (ROOT / '.envs').mkdir(exist_ok=True)
    for link, target in requests:
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        print(f'{link.relative_to(ROOT)} -> {target}')


if __name__ == '__main__':
    main()
