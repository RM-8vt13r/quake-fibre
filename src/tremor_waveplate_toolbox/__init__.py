"""
Initialise the tremor-waveplate-toolbox module by specifying its properties and importing all its functionality.
"""
from .constants import Domain, Device, PAULI_1, PAULI_2, PAULI_3, PAULI_VECTOR
from .constellation import Constellation, BPSK, QPSK, PSK8, QAM4, QAM16, QAM64
from .drift import Drift
from .earthquake import Earthquake
from .fibre_coarse_step import FibreCoarseStep
from .fibre_marcuse import FibreMarcuse
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