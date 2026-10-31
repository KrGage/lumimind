"""Run from the extracted bundle. Dry-run by default; no Git or network needed."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path: Path) -> str:
    # Windows Git may check out CRLF; equivalent source should still match.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def install(bundle: Path, target: Path, apply: bool = False) -> list[str]:
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if not (target / 'backend' / 'app' / 'main.py').is_file():
        raise ValueError('Target must be the agent project root containing backend/app/main.py.')
    changes: list[str] = []
    for relative, entry in manifest.items():
        source = bundle / 'payload' / relative
        destination = target / relative
        if not source.resolve().is_relative_to((bundle / 'payload').resolve()) or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError('Invalid package path: ' + relative)
        if digest(source) != entry['new_hash']:
            raise ValueError('Package checksum mismatch: ' + relative)
        if destination.exists():
            if entry.get('preserve'):
                continue
            current = digest(destination)
            if current == entry['new_hash']:
                continue
            if current not in entry.get('old_hashes', []):
                raise ValueError('Custom/different file detected; nothing was installed: ' + relative
                                 + '. See manual integration instructions; do not force overwrite.')
        elif entry.get('required_existing'):
            raise ValueError('Missing expected agent entry: ' + relative)
        changes.append(relative)
    if not apply:
        return changes
    backup = target / ('ultra_addon_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for relative in changes:
        destination = target / relative
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / 'payload' / relative, destination)
    if backup.exists():
        print('Backup:', backup)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Install after all compatibility checks pass')
    args = parser.parse_args()
    try:
        changes = install(Path(__file__).resolve().parent, args.agent_root.resolve(), args.apply)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(('Installed' if args.apply else 'Preview only') + ': ' + str(len(changes)) + ' files')
    for path in changes:
        print('  ' + path)


if __name__ == '__main__':
    main()
