from distutils.command.config import config
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .adaspeech_modules import UtteranceEncoder, PhonemeLevelEncoder, PhonemeLevelPredictor, Condional_LayerNorm
from utils.tools import get_mask_from_lengths

def get_num_speaker(preprocessed_path):
    f = open(os.path.join(preprocessed_path, 'speakers.json'))
    speakers_dict = json.load(f)
    return len(speakers_dict)
    
class AdaSpeech(nn.Module):
    """ AdaSpeech """

    def __init__(self, preprocess_config, model_config):
        super(AdaSpeech, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.UtteranceEncoder = UtteranceEncoder(model_config)
        self.PhonemeLevelEncoder = PhonemeLevelEncoder(model_config)
        self.PhonemeLevelPredictor = PhonemeLevelPredictor(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.open_set_speaker = preprocess_config['open_set_speaker']
        if not self.open_set_speaker:
            num_speakers = get_num_speaker(preprocess_config["path"]["preprocessed_path"])
            print(num_speakers)
            self.speaker_emb = nn.Embedding(
                num_speakers,
                model_config["transformer"]["encoder_hidden"]
            )
        self.phone_level_embed = nn.Linear(
            model_config["PhoneEmbedding"]["phn_latent_dim"],
            model_config["PhoneEmbedding"]["adim"]
        )
        self.layer_norm = Condional_LayerNorm(preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.postnet = PostNet()

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        avg_targets=None,
        phoneme_level_predictor=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        if not self.open_set_speaker:
            speakers = self.speaker_emb(speakers)
        output = self.encoder(texts, speakers, src_masks)
        xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        xs = torch.transpose(xs, 1, 2)
        output = output + xs.expand(-1, max_src_len, -1)

        if phoneme_level_predictor:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            with torch.no_grad():
                phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            output = output + self.phone_level_embed(phn_encode.detach())
        else:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            output = output + self.phone_level_embed(phn_encode)

        output = output + speakers.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speakers, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speakers)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )

    def inference(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        avg_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        if not self.open_set_speaker:
            speakers = self.speaker_emb(speakers)
        output = self.encoder(texts, speakers, src_masks)
        xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        xs = torch.transpose(xs, 1, 2)
        output = output + xs.expand(-1, max_src_len, -1)

        phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
        phn_encode = None
        output = output + self.phone_level_embed(phn_predict)

        output = output + speakers.unsqueeze(1).expand(
            -1, max_src_len, -1
        )
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speakers, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speakers)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )
        
