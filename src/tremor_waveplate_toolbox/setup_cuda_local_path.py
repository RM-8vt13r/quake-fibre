"""
On older systems or when the system CUDA installation is broken, the CUDA version of the toolbox will not work.
After installing the "cuda_local" version of the toolbox, run this function before importing cupy, to make cupy use the local CUDA environment within this environment.
"""
import os
import ctypes
import logging

logger = logging.getLogger()

def setup_cuda_local_path():
    try:
        import nvidia
        nvidia_local_path = nvidia.__path__[0]
    except:
        raise ImportError("Tried to run setup_cuda_local.py, but the .cuda_local version of the toolbox wasn't installed")

    # Link local CUDA libraries
    targets = {'libnvrtc.so.': False, 'libcudart.so.': False, 'libcurand.so.': False, 'libcusolver.so.': False}
    for dirpath, _, filenames in os.walk(nvidia_local_path):
        for filename in filenames:
            for target in targets:
                if target in filename:
                    dll_path = os.path.join(dirpath, filename)
                    ctypes.CDLL(dll_path)
                    targets[target] = True
                    logger.info(f"Found and added local {filename}")
                    break

    for target, found in targets.items():
        if not found:
            raise ImportError(f"Local {target}* not found in directory {nvidia_local_path}")