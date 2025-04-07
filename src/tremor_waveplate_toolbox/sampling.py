# """
# Signal up- and downsampling
# """
#
# import numpy as np
#
# from .signal import Signal
# import .constellation
#
# # def downsample(signal: Signal, factor: int):
# #     """
# #     Downsample a signal
# #
# #     Inputs:
# #     - signal [Signal]: the signal to downsample
# #     - factor [int]: the downsampling factor
# #
# #     Outputs:
# #     - The downsampled signal
# #     """
# #     assert isinstance(factor, int), f"Downsampling factor must be an integer, but was a {type(factor)}"
# #     assert factor > 1, f"Downsampling factor must be > 1, but was {factor}"
# #
# #     samples_f = signal.samples_f
# #     samples_f[..., np.abs(signal.f) > signal.sample_rate / (2 * factor), :] = 0 # Antialiasing
# #
# #     downsampled_samples = np.fft.ifft(samples_f)[::factor]
# #     downsampled_signal = Signal(downsampled_samples, signal.sample_rate / factor)
# #
# #     return downsampled_signal
#
# def upsample(signal: Signal, factor: int):
#     """
#     Upsample a signal
#
#     Inputs:
#     - signal [Signal]: the signal to upsample
#     - factor [int]: the upsampling factor
#
#     Outputs:
#     - The upsampled signal
#     """
#     assert isinstance(factor, int), f"Upsampling factor must be an integer, but was a {type(factor)}"
#     assert factor > 1, f"Upsampling factor must be > 1, but was {factor}"
#
#     upsampled_samples = np.zeros([*signal.shape[:-2], factor * signal.shape[-2], signal.shape[-1]])
#     upsampled_samples[::factor] = signal.samples
#     upsampled_signal = Signal(upsampled_samples, signal.sample_rate * factor)
#
#     return upsampled_signal
