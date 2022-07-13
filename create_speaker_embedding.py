from multiprocessing.pool import Pool
from functools import partial
from itertools import chain
from speaker_encoder import inference as encoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import argparse
from scipy.io.wavfile import read
import librosa
import yaml

def embed_utterance(wav_fpath, encoder_model_fpath):
    # Compute the speaker embedding of the utterance
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)
    wav, sr = librosa.load(wav_fpath)
    # samplerate, data = read(wav_fpath)
    # wav = np.array(data, dtype=float)
    # wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    return embed
    
def embed_and_save_speaker_vactor(fpaths, speaker_embedding_model_path):
    wav_fpath, embed_fpath = fpaths
    embed = embed_utterance(wav_fpath, speaker_embedding_model_path)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(data_dir: Path, processed_dir: Path, speaker_embedding_model_path: Path, preprocess_config_path: Path, n_processes: int):
    with open(preprocess_config_path) as file:
        preprocess_config = yaml.load(file, Loader=yaml.SafeLoader)
        speaker_embedding_model_path = preprocess_config['path']['speaker_embedding_model_path']
        
    metadata_fpath = data_dir.joinpath("train.txt")
    wav_dir = data_dir.joinpath("train_data")
    assert wav_dir.exists() and metadata_fpath.exists()
    # print(wav_dir)
    embed_dir = processed_dir.joinpath("speaker_embedding")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[1], m[0]+'.wav'), embed_dir.joinpath(m[1]+'-'+m[0]+'.npy')) for m in metadata]  ## m[0] is file name and m[1] is speaker id

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_and_save_speaker_vactor, speaker_embedding_model_path=speaker_embedding_model_path)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file.")
    parser.add_argument("--processed_dir", type=Path, help=\
        "Path to the processed data that contaned (pitch, duration, energy)")
    parser.add_argument("-e", "--speaker_embedding_model_path", type=Path,help="Path your trained encoder model.")
    parser.add_argument("--preprocess_config_path", type=Path, help="Path to preprocess config.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    

    create_embeddings(**vars(args))