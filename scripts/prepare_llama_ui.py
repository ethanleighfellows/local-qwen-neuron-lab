"""Extract the official b11312 embedded web UI via a short-lived local server."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import time
import zipfile

import httpx
from .prepare_llama_libraries import ARCHIVE, ROOT, SHA256

DEST = ROOT / 'data' / 'llama-ui-b11312'
BUNDLE = ROOT / 'data' / 'llama-b11312' / 'ui-extractor'


def main():
    if os.name != 'nt':
        raise RuntimeError('The bundled llama.cpp UI extractor requires Windows')
    if not ARCHIVE.is_file() or hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Verified llama.cpp b11312 archive not found; run install-llama.cmd')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if '/' not in name and (name.endswith('.dll') or name == 'llama-server.exe'):
                (BUNDLE / name).write_bytes(archive.read(name))
    empty_models = BUNDLE / 'empty-models'
    empty_cache = BUNDLE / 'empty-cache'
    empty_models.mkdir(exist_ok=True)
    empty_cache.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env['LLAMA_CACHE'] = str(empty_cache)
    process = subprocess.Popen([str(BUNDLE / 'llama-server.exe'), '--host', '127.0.0.1',
        '--port', str(port), '--models-dir', str(empty_models)], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=3) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError('Official llama.cpp UI extractor exited before serving')
                try:
                    index = client.get('/')
                    index.raise_for_status()
                    break
                except httpx.TransportError:
                    time.sleep(.1)
            else:
                raise RuntimeError('Official llama.cpp UI extractor did not start')
            sw = client.get('/sw.js')
            sw.raise_for_status()
            names = set(re.findall(r'url:"([^"?#]+)"', sw.text))
            # Server-embedded assets are also named in the binary table. This
            # includes non-PWA assets such as recommended tool icons.
            binary = (BUNDLE / 'llama-server-impl.dll').read_bytes()
            names.update(match.decode('ascii') for match in re.findall(
                rb'([A-Za-z0-9_./-]+\.(?:js|css|html|svg|png|ico|json|webmanifest|woff2?|wasm))\x00', binary))
            names.update(('index.html', 'favicon.ico', 'favicon.svg', 'sw.js', 'manifest.webmanifest'))
            DEST.mkdir(parents=True, exist_ok=True)
            count = 0
            for name in sorted(names):
                name = name.removeprefix('./')
                parts = PurePosixPath(name).parts
                if not parts or name.startswith('/') or any(p in ('.', '..') for p in parts):
                    continue
                path = DEST.joinpath(*parts)
                if not path.resolve().is_relative_to(DEST.resolve()):
                    raise RuntimeError('Unexpected asset path')
                response = index if name == 'index.html' else client.get('/' + name)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                count += 1
            if not (DEST / 'index.html').is_file() or not list(DEST.glob('_app/immutable/*.js')):
                raise RuntimeError('Official web UI assets are incomplete')
            print(f'Prepared {count} official llama.cpp web UI assets in {DEST}')
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
