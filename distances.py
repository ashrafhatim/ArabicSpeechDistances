"""
Main Audio distance computation module.
"""

import os
import time
import numpy as np
from glob import glob
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from utils import *
from exceptions import CustomError
# from preprocessing import get_speech_frames

from dataset import custom_dataset
from torch.utils.data import DataLoader

import torch

from transformers import (Wav2Vec2ForCTC,Wav2Vec2Processor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDistance(object):
    """Main DeepSpeech Distance evaluation class."""

    def __init__(self, model_dir='elgeish/wav2vec2-large-xlsr-53-arabic', max_length=128000, sr=16000, gpu_id=0):
        self.model_dir = model_dir
        self.max_length = max_length
        self.sr = sr
        self.gpu_id=gpu_id

        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_dir).to(device)
        self.model.wav2vec2.feature_projection.register_forward_hook(self.output_hook)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_dir)

        self.activation = {}

    def output_hook(self, model, input, output):
        self.activation["norm_hidden_states"] = output[1].detach()
        raise CustomError

    def get_activations(self, data):
        input_features = self.processor(data[:,:self.max_length],
                            sampling_rate=self.sr,
                            padding=True,
                            max_length=self.max_length, 
                            pad_to_multiple_of=None,
                            return_tensors="pt")

        input_values = input_features.input_values
        input_values.squeeze_(0)
        input_values = to_cuda(input_values, self.gpu_id)

        try: _ = self.model(input_values)
        except CustomError: pass

        out = torch.nn.functional.adaptive_avg_pool2d(self.activation["norm_hidden_states"], (40,40) )
        out = out.view(data.shape[0], 1600)

        return out

    # def get_features(self, audio):
    def get_features(self, frames):
        # frames = get_speech_frames(audio, sample_freq = 16000, window_size=40e-3,
        #                 window_stride=20e-3)
        activations = self.get_activations(frames)
        features = activations.sum(axis=0)/activations.size()[0]
        return features

    def get_distribution(self, samples_path):
        paths = glob(samples_path + "/*")


        test_set = custom_dataset(paths)
        test_loader = DataLoader(test_set, batch_size=1)
        
        distribution = torch.empty((len(paths), 1600))
        # for i,path in enumerate(tqdm(paths)):
        for i,frames in enumerate(test_loader):
            # audio = load_file_to_data(path, srate = 22050)["speech"]
            # features = self.get_features(audio)
            
            frames.squeeze_(0)
            features = self.get_features(frames)
            distribution[i, :] = features    

        return distribution

    # def stats(self, distribution):
    #     mu = torch.mean(distribution, axis=0)
    #     sigma = torch.cov(distribution.T)
    #     return mu, sigma.numpy()






