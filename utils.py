"""Utility file to subsample random clips from longer audio file."""

from scipy.io.wavfile import read, write
import os
import numpy as np
from tqdm import tqdm

import librosa    
import torch

import warnings
from scipy import linalg


import torch
import io
import os
import scipy.io.wavfile
import numpy as np
# import tensorflow.compat.v1 as tf
import resampy as rs
import python_speech_features as psf


def _mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def subsample_audio(file, sample_path, save_file_name, sample_num, num_samples=1000,
                    num_noise_levels=3, length=2, freq = None):
  """Helper sampling function.

  Args:
    file_path: Path to the source audio file.
    sample_path: Path to save subsamples in.
    num_samples: Numer of clips to sample from the source file.
    num_noise_levels: Number of noise levels. Apart from clean samples,
       this number of noisy versions of each sampled clip will be saved, with
       noise levels chosen from logspace between 10^-3 and 10^-1.5.
    length: Length of subsampled clips, in seconds.
  """
  if type(file) == str:
    freq, base_wav = read(file)
  elif freq != None:
    freq, base_wav = freq, file
    
  base_wav = base_wav.astype(np.float32) / 2**15
  length *= freq

  # start = np.random.randint(0, base_wav.shape[0] - length + 1,
  #                           size=(2 * num_samples, ))

  start = np.linspace(0, base_wav.shape[0] - length + 1, num_samples)

  # noise_levels = np.logspace(-3, -1, num_noise_levels)

  _mkdir(sample_path)
  _mkdir(os.path.join(sample_path, save_file_name))
  # for i in range(num_noise_levels):
  #   _mkdir(os.path.join(sample_path, f'noisy_{i + 1}'))

  for k, start_k in enumerate(start):
    start_k = int(start_k)
    window = base_wav[start_k: start_k + length]
    write(os.path.join(sample_path, save_file_name, '%03d_%05d.wav' % (sample_num, (k + 1))), freq, window)

    # for i, noise_level in enumerate(noise_levels):
    #   noisy_window = window + np.random.normal(scale=noise_level, size=(length, ))
    #   noisy_window = np.clip(noisy_window * 2 ** 15, -2 ** 15, 2 ** 15 - 1)
    #   write(os.path.join(sample_path, f'noisy_{i + 1}', '%05d.wav' % (k + 1)),
    #         freq, noisy_window.astype(np.int16))

def normalize_signal(signal):
  """
  Normalize float32 signal to [-1, 1] range
  """
  return signal / (np.max(np.abs(signal)) + 1e-5)


def get_speech_frames(signal,
                        sample_freq=22050 ,
                        num_features=160,
                        pad_to=8,
                        window_size=20e-3,
                        window_stride=10e-3,
                        base_freq=16000):
  """Function to convert raw audio signal to array of frames.
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (int): Frames per second.
    num_features (int): Number of speech features in frequency domain.
    pad_to (int): If specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    window_size (float): Size of analysis window in milli-seconds.
    window_stride (float): Stride of analysis window in milli-seconds.
    base_freq (int): Frequency at which spectrogram will be computed.

  Returns:
    tensor of audio frames.
  """
  signal = signal.astype(np.float32)

  if sample_freq != base_freq:
    signal = rs.resample(signal, sample_freq, base_freq, filter='kaiser_best')
    sample_freq = base_freq
  
  signal = normalize_signal(signal)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

  length = 1 + int(np.ceil(
      (1.0 * signal.shape[0] - n_window_size) / n_window_stride))
  if pad_to > 0:
    if length % pad_to != 0:
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='constant')

  frames = psf.sigproc.framesig(sig=signal,
                                frame_len=n_window_size,
                                frame_step=n_window_stride,
                                winfunc=np.hanning)
  
  if num_features > n_window_size // 2 + 1:
    raise ValueError(
       f"num_features (= {num_features}) for spectrogram should be <= (sample_"
       f"freq (= {sample_freq}) * window_size (= {window_size}) // 2 + 1)")
  
  return torch.tensor(frames)



def load_file_to_data(file, srate = 16_000):
    batch = {} 
    speech, sampling_rate = librosa.load(file, sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch

def to_cuda(elements, gpu_id):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda(gpu_id)
    return elements

# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def stats(distribution):
        mu = torch.mean(distribution, axis=0)
        sigma = torch.cov(distribution.T)
        return mu, sigma.numpy()

