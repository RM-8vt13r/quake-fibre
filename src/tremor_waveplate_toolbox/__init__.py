"""
Initialise the tremor-waveplate-toolbox module by specifying its properties and importing all its functionality.
"""
from .fibre import Fibre
from .signal import Signal
from .transmitter import Transmitter
from .receiver import Receiver
from .pulse import Pulse, Sinc, SINC, RootRaisedCosine, RRCOS
from .constellation import Constellation, BPSK, QPSK, PSK8, QAM4, QAM16, QAM64
from .constants import Domain, PAULI_1, PAULI_2, PAULI_3, PAULI_VECTOR
from .earthquake import Earthquake