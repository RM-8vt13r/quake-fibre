"""
A base class for various optical fibre channel models for dual-polarisation transmission.
"""

from configparser import ConfigParser
from abc import ABC, abstractmethod

from .signal import Signal

class Channel(ABC):
    """
    An abstract class representing a channel, over which a signal can be propagated.
    """
    def __init__(self, parameters: ConfigParser):
        """
        Create the channel.

        Inputs:
        - parameters [ConfigParser]: the channel parameters, loaded from a .ini file.
        """
        pass

    def __call__(self, signal: Signal) -> Signal:
        """
        Make channel instances callable; see propagate()
        """
        return self.propagate(signal)

    @abstractmethod
    def propagate(self, signal: Signal) -> Signal:
        """
        Propagate a signal through this channel.

        Inputs:
        - signal [Signal]: the signal to propagate through the channel, shape [B,N,Q,P] with batch size B, signal length N, in-phase- and quadrature components Q=2, and principal polarisations P=2.

        Outputs:
        - [Signal]: the output signal, shape [B,N,Q,P]
        """
        raise NotImplementedError()
