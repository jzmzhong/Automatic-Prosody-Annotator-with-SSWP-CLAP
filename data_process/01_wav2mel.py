import os
from tqdm import tqdm
import yaml
import numpy as np
from audios.dsp import Wav2Mel


"""
The directory hierarchy of data inputs
├── internal-spk1-test
│   └── audios
│       ├── 1.wav
│       └── 2.wav

"""

DATA_ROOT = "../sample_data/01_input_data"
ITEMS = ["internal-spk1-test"]

CONFIG_PATH = "./audios/TTS_CLAP_16k.yaml"

class AttributeDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def load_yaml(file_path):
    with open(file_path) as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)
        attribute_dict = AttributeDict(loaded_config)
    return attribute_dict


def main():
    global item
    config = load_yaml(CONFIG_PATH)
    print(config)
    wav2mel = Wav2Mel(config)
    for item in ITEMS:
        in_folder = os.path.join(DATA_ROOT, item, "audios")
        out_folder = os.path.join(DATA_ROOT, item, "mels")
        os.makedirs(out_folder, exist_ok=True)
        wav_names = [_ for _ in os.listdir(in_folder) if _.endswith(".wav")]
        print("Processing", in_folder)
        for wav_name in tqdm(wav_names):
            wav_path = os.path.join(in_folder, wav_name)
            mel_path = os.path.join(out_folder, wav_name.replace(".wav", ".mel.npy"))
            try:
                mel, wav, pitch, energy = wav2mel(wav_path)
                np.save(mel_path, mel)
            except:
                print("Processing Error", wav_name)


if __name__ == '__main__':
    main()
