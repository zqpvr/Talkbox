"""Windows CUDA DLL bootstrapping.

Two separate problems are solved here, both of which abort the process on a
stock install.

First, CTranslate2 (the engine under faster-whisper) links cuBLAS and cuDNN by
name and relies on the OS loader to find them. When CUDA comes from pip wheels
rather than a system CUDA Toolkit, those DLLs live inside site-packages under
torch/lib and nvidia/*/bin, neither of which is searched by default. The
symptom is a blunt "Library cublas64_12.dll is not found" at first model load.

Second, and nastier: cuDNN 9 is split into a small dispatcher (cudnn64_9.dll)
plus large sub-libraries (cudnn_graph64_9.dll, cudnn_ops64_9.dll and friends).
The ctranslate2 wheel bundles *only* the dispatcher, from a newer cuDNN than
the complete set torch bundles. Because a DLL's own directory is searched
first, ctranslate2 gets its own newer dispatcher, which then binds against
torch's older sub-libraries and dies with "Could not load symbol
cudnnGetLibConfig. Error code 127" — a hard process abort, not an exception.

The fix is to preload torch's matched cuDNN set by absolute path before
anything imports ctranslate2. Windows resolves subsequent LoadLibrary calls for
an already-loaded module name to the loaded instance, so the whole process ends
up sharing one internally consistent cuDNN instead of interleaving two.
"""

from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

_CANDIDATE_SUBDIRS = (
    "cublas/bin",
    "cudnn/bin",
    "cuda_runtime/bin",
    "cuda_nvrtc/bin",
    "cufft/bin",
)


def _package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except AttributeError:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    # Only consult the per-user site directory outside a virtual environment.
    # Inside one, Python already ignores it for imports, and registering its
    # torch/lib would put a second, possibly mismatched, set of cuDNN and
    # cuBLAS DLLs on the loader's search path ahead of the venv's own.
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        try:
            user = site.getusersitepackages()
            roots.append(Path(user) if isinstance(user, str) else Path(user[0]))
        except Exception:
            pass

    return [r for r in dict.fromkeys(roots) if r.is_dir()]


# Order matters: every sub-library must be resident before the dispatcher, so
# that the dispatcher's own lookups resolve to this set rather than to whatever
# another wheel put on the path.
_CUDNN_PRELOAD = (
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn64_9.dll",
)


def _preload_matched_cudnn(roots: list[Path]) -> list[str]:
    """Pin one complete cuDNN installation into the process.

    Prefers a standalone nvidia-cudnn-cu12 wheel if present, since that is a
    known-complete set; otherwise falls back to the copy torch ships, which is
    also complete. What must be avoided is the ctranslate2 directory, which
    holds a lone dispatcher with no sub-libraries behind it.
    """
    for root in roots:
        for candidate in (root / "nvidia" / "cudnn" / "bin", root / "torch" / "lib"):
            if not (candidate / "cudnn64_9.dll").is_file():
                continue
            if not (candidate / "cudnn_graph64_9.dll").is_file():
                continue  # Dispatcher without sub-libraries; skip it.
            loaded: list[str] = []
            for name in _CUDNN_PRELOAD:
                path = candidate / name
                if not path.is_file():
                    continue
                try:
                    ctypes.WinDLL(str(path))
                    loaded.append(name)
                except OSError:
                    pass
            if loaded:
                return loaded
    return []


def ensure_cuda_dlls() -> list[str]:
    """Add pip-installed CUDA library directories to the DLL search path."""
    if os.name != "nt":
        return []

    roots = _package_roots()
    added: list[str] = []
    for root in roots:
        nvidia = root / "nvidia"
        targets: list[Path] = []
        if nvidia.is_dir():
            targets.extend(nvidia / sub for sub in _CANDIDATE_SUBDIRS)
            # Layouts vary between wheel versions; sweep for any other bin dirs.
            targets.extend(p for p in nvidia.glob("*/bin") if p.is_dir())
        # torch/lib goes last: a dedicated nvidia wheel, when installed, is the
        # newer and more complete of the two.
        targets.append(root / "torch" / "lib")

        for target in targets:
            if not target.is_dir():
                continue
            key = str(target)
            if key in added:
                continue
            try:
                os.add_dll_directory(key)
            except (OSError, AttributeError):
                continue
            added.append(key)

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")

    _preload_matched_cudnn(roots)
    return added
