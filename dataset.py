from torch.utils.data import Dataset
from utils import get_speech_frames, load_file_to_data


class custom_dataset(Dataset):
    def __init__(self, paths) -> None:
        super().__init__()
        self.paths=paths
        # self.audios=[ load_file_to_data(path, srate = 22050)["speech"]  for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio = load_file_to_data(path, srate = 22050)["speech"]

        # audio = self.audios[idx]

        frames = get_speech_frames(audio, sample_freq = 16000, window_size=40e-3,
                        window_stride=20e-3)
        return frames

