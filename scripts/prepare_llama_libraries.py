"""Prepare pinned Windows llama.cpp DLLs using the official AVX2 CPU build."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / 'data' / 'llama-b11312' / 'llama-b11312-bin-win-cpu-x64.zip'
DEST = ROOT / 'data' / 'llama-libs-b11312-cu125-avx2'
URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b11312/llama-b11312-bin-win-cpu-x64.zip'
SHA256 = 'd2d966c23e4d097d70633f9d92e4931245b575be7a5409201d8f352493f0e702'


def main():
    if os.name != 'nt':
        raise RuntimeError('This prepared DLL bundle targets Windows x64')
    if importlib.metadata.version('llama-cpp-python') != '0.3.36':
        raise RuntimeError('Install the pinned llama-cpp-python 0.3.36 CUDA wheel first')
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(URL, ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError('Official llama.cpp archive SHA-256 does not match')
    package_root = Path(importlib.metadata.distribution('llama-cpp-python').locate_file('llama_cpp/lib'))
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cuda.dll', 'mtmd.dll'):
        shutil.copy2(package_root / name, DEST / name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        # Exact file names only; no untrusted archive paths are extracted.
        (DEST / 'ggml-cpu.dll').write_bytes(archive.read('ggml-cpu-haswell.dll'))
        (DEST / 'libomp.dll').write_bytes(archive.read('libomp.dll'))
    print(f'Prepared llama.cpp CUDA + official AVX2 CPU libraries in {DEST}')


if __name__ == '__main__':
    main()
