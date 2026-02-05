"""
Initialise the tremor-waveplate-toolbox module by specifying its properties and importing all its functionality.
"""
def _setup_cuda_local_path():
    import os
    import ctypes

    try:
        import nvidia
        nvidia_local_path = nvidia.__path__[0]
    
        # Link local CUDA libraries
        targets = {'libcudart.so.': False, 'libnvrtc.so.': False, 'libcublas.so': False, 'libcufft.so': False, 'libcusolver.so.': False, 'libcusparse.so.': False, 'libcurand.so.': False}
        for dirpath, _, filenames in os.walk(nvidia_local_path):
            for filename in filenames:
                for target in targets:
                    if target in filename:
                        dll_path = os.path.join(dirpath, filename)
                        ctypes.CDLL(dll_path)
                        targets[target] = True
                        break

        # for target, found in targets.items():
        #     if not found:
        #         raise ImportError(f"Local CUDA installation file {target}* not found in directory {nvidia_local_path}")
    except:
        pass

_setup_cuda_local_path()
del _setup_cuda_local_path

from .constants import Domain, Device, PAULI_1, PAULI_2, PAULI_3, PAULI_VECTOR
from .constellation import Constellation, BPSK, QPSK, PSK8, QAM4, QAM16, QAM64
from .drift import Drift
from .earthquake import Earthquake
from .fibre_coarse_step import FibreCoarseStep
from .fibre_cnlse import FibreCNLSE
from .fibre import Fibre
from .path import Path
from .perturbation_event import PerturbationEvent
from .perturbation import Perturbation
from .pulse import Pulse, Sinc, SINC, RootRaisedCosine, RRCOS, Square, SQUARE
from .receiver import Receiver
from .scramblers import Scramblers
from .signal import Signal
from .transmitter import Transmitter
from .utils import rotation_matrix, phase_matrix