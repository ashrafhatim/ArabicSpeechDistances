"""Utility file to subsample random clips from longer audio file."""

from scipy.io.wavfile import read, write
import os
import numpy as np
from tqdm import tqdm

import librosa    
import torch

import warnings
from scipy import linalg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def subsample_audio(file_path, sample_path, num_samples=1000,
                    num_noise_levels=3, length=2):
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
  freq, base_wav = read(file_path)
  base_wav = base_wav.astype(np.float32) / 2**15
  length *= freq

  start = np.random.randint(0, base_wav.shape[0] - length + 1,
                            size=(2 * num_samples, ))
  noise_levels = np.logspace(-3, -1, num_noise_levels)

  _mkdir(sample_path)
  _mkdir(os.path.join(sample_path, 'ref'))
  for i in range(num_noise_levels):
    _mkdir(os.path.join(sample_path, f'noisy_{i + 1}'))

  for k, start_k in enumerate(tqdm(start, desc='Saving audio sample files')):
    window = base_wav[start_k: start_k + length]
    write(os.path.join(sample_path, 'ref', '%05d.wav' % (k + 1)), freq, window)

    for i, noise_level in enumerate(noise_levels):
      noisy_window = window + np.random.normal(scale=noise_level, size=(length, ))
      noisy_window = np.clip(noisy_window * 2 ** 15, -2 ** 15, 2 ** 15 - 1)
      write(os.path.join(sample_path, f'noisy_{i + 1}', '%05d.wav' % (k + 1)),
            freq, noisy_window.astype(np.int16))


def load_file_to_data(file, srate = 16_000):
    batch = {} 
    speech, sampling_rate = librosa.load(file, sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch

def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()
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


