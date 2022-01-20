import argparse
# from distances import AudioDistance
import sys

# imports

from glob import glob
from tqdm import tqdm
import os
import torch
from pathlib import Path

from librosa.core import load
from librosa.util import normalize
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


sys.path.insert(0,'/home/jupyter/ArabicSpeechDistances')
from distances import AudioDistance
from utils import *
# model = AudioDistance()

sys.path.insert(0,'/home/jupyter/melgan-neurips-custom')
from mel2wav.modules import Generator, Discriminator, Audio2Mel

# generator = Generator(80, 32, 3).cuda().eval()
# fft = Audio2Mel(n_mel_channels=80).cuda()


def load_wav_to_torch(full_path, sampling_rate):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = load(full_path, sr=sampling_rate)
    # data = 0.95 * normalize(data)

    # if True:
    #     amplitude = np.random.uniform(low=0.3, high=1.0)
    #     data = data * amplitude

    return torch.from_numpy(data).float(), sampling_rate


def get_distance(file_paths, sample_path, save_file_name, num_samples, length, mu_0=None, sigma_0=None, model_path=None, gpu_id=0):
    
    model = AudioDistance(gpu_id=gpu_id)

    generator = Generator(80, 32, 3).cuda(gpu_id).eval()
    fft = Audio2Mel(n_mel_channels=80).cuda(gpu_id)

    def generate_sample(file_path, fft, generator):
        audio, sampling_rate = load_wav_to_torch(file_path, 22050)
        segment_length = 8192
        audio.unsqueeze(0)

        x_t = audio.unsqueeze(0).cuda(gpu_id)
        s_t = fft(x_t).detach()
        x_pred_t = generator(s_t).cpu().detach().numpy()

        return x_pred_t, sampling_rate
    
    # subsample audios
    if model_path == None:
        # print("subsample the ground truth")
        # for sample_num, file_path in enumerate(tqdm(file_paths)):
        for sample_num, file_path in enumerate(file_paths):
            subsample_audio(file_path, sample_path, save_file_name, sample_num=sample_num+1, num_samples=num_samples, length=length)
    else:
        generator.load_state_dict(torch.load(model_path, map_location='cuda:%d' % gpu_id))
        # print("subsample the generated audios")
        # for sample_num, file_path in enumerate(tqdm(file_paths)):
        for sample_num, file_path in enumerate(file_paths):
            x_pred_t, sampling_rate = generate_sample(file_path, fft, generator)
            subsample_audio(x_pred_t, sample_path, save_file_name, sample_num=sample_num+1, num_samples=num_samples, length=length, freq=sampling_rate)
            
    # # caculate the distibution of ref
    # ref_distribution = model.get_distribution(os.path.join(sample_path, "ref"))
    # mu_0, sigma_0 = stats(ref_distribution)
    
    # caculate the distibution of generated samples
    distribution = model.get_distribution(os.path.join(sample_path, save_file_name))
    mu, sigma = stats(distribution)
    
    # return mu, sigma
    
    # distance
    dist = calculate_frechet_distance(mu_0, sigma_0, mu, sigma)
    
    return dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths", type=list, default=glob("/home/jupyter/data/arabic-speech-corpus/test set/wav/*.wav"))
    parser.add_argument("--sample_path", default="/home/jupyter/data/arabic-speech-corpus")
    parser.add_argument("--save_file_name", default="generated")

    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--exp_path", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # print("agument: ", args.augment)
    root = Path(args.exp_path)
    writer = SummaryWriter(str(root / "FID_tensorboard"))
                        
                        
    sys.path.insert(0,'/home/jupyter/ArabicSpeechDistances')
    ref_stats = torch.load( "ref_stats.pt")
    mu_0 = ref_stats["mu"]
    sigma_0 = ref_stats["sigma"]
    
    
    for i in tqdm(range(5000, 500000, 5000)):
        model_path = root / ("netG_%d.pt" % i)
                             
        distance = get_distance(
            file_paths = args.file_paths,
            sample_path = args.sample_path,
            save_file_name = args.save_file_name,
            num_samples = args.num_samples,
            length = args.length,
            mu_0=mu_0, 
            sigma_0=sigma_0, 
            model_path=model_path,
            gpu_id=args.gpu_id
        )
        
        
        writer.add_scalar("frechetDist/conditional", distance, i/5000)
    
    
    

if __name__ == "__main__":
    main()
