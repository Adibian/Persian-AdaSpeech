import argparse

import torch
import yaml
import sys
import json
import librosa
import numpy as np
import time

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, AttrDict
from dataset import Dataset
from text import text_to_sequence
from datetime import datetime
from g2p_en import G2p

import audio as Audio

sys.path.append("vocoder")
from models.hifigan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vocoder_checkpoint_path = "data/g_02519517"
# vocoder_config = "data/config_22k.json"

# def get_vocoder(config, checkpoint_path):
#     config = json.load(open(config, 'r', encoding='utf-8'))
#     config = AttrDict(config)
#     checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
#     vocoder = Generator(config).to(device).eval()
#     vocoder.load_state_dict(checkpoint_dict['generator'])
#     vocoder.remove_weight_norm()

#     return vocoder

def synthesize(model, step, configs, vocoder, loader, control_values, output_dir):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model.inference(
                *(batch[1:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                output_dir,
            )

def get_reference_mel(reference_audio_dir, STFT):
    wav, _ = librosa.load(reference_audio_dir)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
    return mel_spectrogram

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    
    parser.add_argument(
        "--output_dir",
        default='output/result',
        type=str
    )

    parser.add_argument(
        "--ref_utterance",
        type=str
    )

    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )

    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open("config/pretrain/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open("config/pretrain/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/pretrain/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    output_dir = args.output_dir
    wav_path = args.ref_utterance
    STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    # vocoder = get_vocoder(vocoder_config, vocoder_checkpoint_path)
    vocoder = get_vocoder(model_config, device)
    
    # Preprocess texts
    t = str(time.time()).split('.')[0]
    ids = [str(args.ref_utterance).split('/')[-1].split('.')[0] + t]
    speakers = np.array([args.ref_utterance])
    texts = np.array([text_to_sequence(args.text, preprocess_config['preprocessing']['text']['text_cleaners'])])
    text_lens = np.array([len(texts[0])])
    mel_spectrogram = get_reference_mel(wav_path, STFT)
    mel_spectrogram = np.array([mel_spectrogram])
    batchs = [(ids, speakers, texts, text_lens, max(text_lens), mel_spectrogram)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, output_dir)
