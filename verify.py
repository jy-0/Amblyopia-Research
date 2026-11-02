"""Verify the SHA256 of every file in the published release manifest."""
import json
from pathlib import Path
from core import sha256


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest['files']:
        path = root / item['path']
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise RuntimeError(f'File checksum mismatch: {item["path"]}')
    print(f'Verified {len(manifest["files"])} release files.')


if __name__ == '__main__':
    main()
