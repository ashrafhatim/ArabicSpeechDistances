

# import torch
# import io
# import os
# import scipy.io.wavfile
# import numpy as np
# # import tensorflow.compat.v1 as tf
# import resampy as rs
# import python_speech_features as psf


# def normalize_signal(signal):
#   """
#   Normalize float32 signal to [-1, 1] range
#   """
#   return signal / (np.max(np.abs(signal)) + 1e-5)


# def get_speech_frames(signal,
#                         sample_freq,
#                         num_features=160,
#                         pad_to=8,
#                         window_size=20e-3,
#                         window_stride=10e-3,
#                         base_freq=16000):
#   """Function to convert raw audio signal to array of frames.
#   Args:
#     signal (np.array): np.array containing raw audio signal.
#     sample_freq (int): Frames per second.
#     num_features (int): Number of speech features in frequency domain.
#     pad_to (int): If specified, the length will be padded to become divisible
#         by ``pad_to`` parameter.
#     window_size (float): Size of analysis window in milli-seconds.
#     window_stride (float): Stride of analysis window in milli-seconds.
#     base_freq (int): Frequency at which spectrogram will be computed.

#   Returns:
#     tensor of audio frames.
#   """
#   signal = signal.astype(np.float32)

#   if sample_freq != base_freq:
#     signal = rs.resample(signal, sample_freq, base_freq, filter='kaiser_best')
#     sample_freq = base_freq
  
#   signal = normalize_signal(signal)

#   audio_duration = len(signal) * 1.0 / sample_freq

#   n_window_size = int(sample_freq * window_size)
#   n_window_stride = int(sample_freq * window_stride)

#   length = 1 + int(np.ceil(
#       (1.0 * signal.shape[0] - n_window_size) / n_window_stride))
#   if pad_to > 0:
#     if length % pad_to != 0:
#       pad_size = (pad_to - length % pad_to) * n_window_stride
#       signal = np.pad(signal, (0, pad_size), mode='constant')

#   frames = psf.sigproc.framesig(sig=signal,
#                                 frame_len=n_window_size,
#                                 frame_step=n_window_stride,
#                                 winfunc=np.hanning)
  
#   if num_features > n_window_size // 2 + 1:
#     raise ValueError(
#        f"num_features (= {num_features}) for spectrogram should be <= (sample_"
#        f"freq (= {sample_freq}) * window_size (= {window_size}) // 2 + 1)")
  
#   return torch.tensor(frames)
