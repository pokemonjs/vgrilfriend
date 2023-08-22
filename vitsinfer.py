import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser
import sys
sys.path.append('E:/StudyWork/V/VITS-barbara')
os.chdir('E:/StudyWork/V/VITS-barbara')

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./OUTPUT_MODEL/G_latest.pth", help="directory to your fine-tuned model")
parser.add_argument("--config_dir", default="./configs/modified_finetune_speaker.json", help="directory to your model config file")
parser.add_argument("--share", default=False, help="make link public (used in colab)")
parser.add_argument("--text", default='''可以看到，我首先问了然后问“为什么是这样”，ChatGPT会根据前面的提问将新问题识别为“为什么1+1=2”。''', help="directory to your model config file")

args = parser.parse_args()
print("args:",args.config_dir,args.model_dir)
hps = utils.get_hparams_from_file(args.config_dir)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()
_ = utils.load_checkpoint(args.model_dir, net_g, None)
speaker_ids = hps.speakers
print(speaker_ids)
speakers = list(hps.speakers.keys())
tts_fn = create_tts_fn(net_g, hps, speaker_ids)
vc_fn = create_vc_fn(net_g, hps, speaker_ids)

import gradio as gr
audio_gr = gr.Audio(label="Output Audio", elem_id="tts-audio")

from gradio import processing_utils

def infer(text,speed=0.5):
    text_output, audio_output = tts_fn(text,speakers[0],lang[1],speed)
    sample_rate, data = audio_output
    file_path = audio_gr.audio_to_temp_file(data, sample_rate, format="mp3")
    processing_utils.audio_to_file(sample_rate, data, "out.wav")
    print(file_path)
    return file_path
    # print(text_output)
    # print(type(audio_output))
    # print(type(audio_output[0]),audio_output[0])
    # print(type(audio_output[1]),audio_output[1])


test_text = '''可以看到，我首先问了“1+1=几”，然后问“为什么是这样”，ChatGPT 会根据前面的提问将新问题识别为“为什么1+1=2”。
后面继续问水仙花数有哪些，再问“如何写个python程序来识别这些数”，ChatGPT 同样会根据前面的提问将新问题识别为“如何写个python程序来识别这些水仙花数”，并给出对应解答。'''
print(args.text)
infer(args.text,0.5)