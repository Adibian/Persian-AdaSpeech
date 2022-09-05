import os
import json

import torch
import numpy as np

from hifigan import En_HiFiGAN
from model import AdaSpeech, ScheduledOptim
from parallel_wavegan.utils import load_model

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = AdaSpeech(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def load_pretrain(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = AdaSpeech(preprocess_config, model_config).to(device)
    ckpt_path = args.pretrain_dir
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    scheduled_optim = ScheduledOptim(
        model, train_config, model_config, 0
    )
    for param in model.named_parameters():
        if "layer_norm" not in param[0]:
            param[1].requires_grad = False
        if "encoder" in param[0]:
            param[1].requires_grad = False
        if "variance_adaptor" in param[0]:
            param[1].requires_grad = False
        if "UtteranceEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelPredictor" in param[0]:
            param[1].requires_grad = False
        if "speaker_emb" in param[0]:
            param[1].requires_grad = True
    model.train()
    return model, scheduled_optim
    

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "En_HiFiGAN":
        with open("hifigan/En_HiFiGAN/config.json", "r") as f:
            config = json.load(f)
        config = En_HiFiGAN.AttrDict(config)
        vocoder = En_HiFiGAN.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/En_HiFiGAN/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/En_HiFiGAN/generator_universal.pth.tar", map_location=torch.device(device))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    
    elif name == "Persian_HiFiGAN":
        vocoder = load_model("hifigan/Persian_HiFiGAN/hifigan_ch200000_v2.pkl")
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
        
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "En_HiFiGAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "Persian_HiFiGAN":
            mels_for_hifigan = mels.squeeze().T
            wavs = vocoder.inference(mels_for_hifigan).T
    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs