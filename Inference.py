"""
Usage: env CUDA_VISIBLE_DEVICES=2 python -m Inference
"""

import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import yaml
import torch
import numpy as np
from scipy.io import wavfile

from Modules import HifiSinger
from Arg_Parser import Recursive_Parse
from Datasets import Duration_Stack, Note_Stack, Token_Stack

kInferenceAcousticModelCheckpointPath = \
    "/home/data/futabanzu/old/hifisinger/checkpoint_/S_400000.pt"
kInferenceVocoderModelPath = \
    "/home/data/futabanzu/hifisinger/Vocoder.pts"
kInferenceHyperParametersPath = \
    "/home/lab/futabanzu/HIFISinger/kiritan/hp_hifisinger.yaml"
kInferenceTokenDictPath = "/home/lab/futabanzu/HIFISinger/kiritan/tokens.yaml"
kInferenceInputLabelPath = \
    "/home/lab/futabanzu/HIFISinger/tmp/waseda_label.txt"
kInferenceOutputWavefilePath = \
    "/home/lab/futabanzu/HIFISinger/tmp/output_no_aug_waseda.wav"
kInferenceOutputPlotPath = kInferenceOutputWavefilePath + ".plot.png"


class Inference:
    def __init__(
        self,
        acoustic_model_path: str,
        vocoder_model_path: str,
        hyperparameter_path: str,
        tokens_path: str,
    ):
        self.hp = Recursive_Parse(yaml.load(
            open(hyperparameter_path, encoding='utf-8'),
            Loader=yaml.Loader
        ))
        self.tokens: Dict[str, str] = yaml.load(
            open(tokens_path, 'r'), Loader=yaml.Loader)

        self.acoustic_model_path = acoustic_model_path
        self.vocoder_model_path = vocoder_model_path

        self.device = self.__load_device()

        self.__load_model()

    def __load_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device('cpu')

        self.gpu_id = int(os.getenv('RANK', '0'))

        device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)

        return device

    def __load_model(self):
        self.model: HifiSinger = HifiSinger(self.hp).to(self.device)
        self.model.requires_grad_(False)

        checkpoint = torch.load(self.acoustic_model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['Generator']['Model'])

        self.vocoder = torch.jit.load(self.vocoder_model_path).to(self.device)

    def __call__(
            self,
            label_path: str,
            output_wav_path: str,
            output_plot_path: str):
        music = self.load_mono_label(label_path)
        duration: Tuple[float]
        lyric: Tuple[str]
        note: Tuple[int]
        duration, lyric, note = zip(*music)

        durations = Duration_Stack([duration])
        tokens = Token_Stack([lyric], self.tokens)
        notes = Note_Stack([note])
        token_lengths = [len(lyric) + 1]

        durations = torch.LongTensor(durations)
        tokens = torch.LongTensor(tokens)
        notes = torch.LongTensor(notes)
        token_lengths = torch.LongTensor(token_lengths)

        mels, silences, pitches, durations = self.__infer_internal(
            durations, tokens, notes, token_lengths)

        self.__plot_result(mels, silences, pitches,
                           durations, output_plot_path)

        mel = mels[0]
        silence = silences[0]
        pitch = pitches[0]
        duration = durations[0]

        self.__infer_vocoder_internal(mel, silence, pitch, output_wav_path)

    @torch.no_grad()
    def __infer_vocoder_internal(
            self,
            mel,
            silence,
            pitch,
            output_filename: str):
        if self.vocoder is None:
            raise Exception('vocoder is not available')

        mel = mel.unsqueeze(0)
        silence = silence.unsqueeze(0)
        pitch = pitch.unsqueeze(0)

        x = torch.randn(
            size=(
                mel.size(0),
                self.hp.Sound.Frame_Shift * mel.size(2))
        ).to(self.device)
        mel = torch.nn.functional.pad(mel, (2, 2), 'reflect').to(self.device)
        silence = torch.nn.functional.pad(silence.unsqueeze(
            dim=1), (2, 2), 'reflect').squeeze(dim=1).to(self.device)
        pitch = torch.nn.functional.pad(pitch.unsqueeze(
            dim=1), (2, 2), 'reflect').squeeze(dim=1).to(self.device)

        wav = self.vocoder(x, mel, silence, pitch).cpu().numpy()[0]
        wavfile.write(
            filename=output_filename,
            data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7)
                  * 32767.5).astype(np.int16),
            rate=self.hp.Sound.Sample_Rate
        )

    @torch.no_grad()
    def __infer_internal(
            self,
            durations: torch.Tensor,
            tokens: torch.Tensor,
            notes: torch.Tensor,
            token_lengths: torch.Tensor) -> Tuple[
                torch.LongTensor,
                torch.LongTensor,
                torch.LongTensor,
                torch.LongTensor]:
        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)

        mels, silences, pitches, durations = self.model(
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths
        )

        return mels, silences, pitches, durations

    def load_mono_label(self, label_path: str) -> List[Tuple[float, int, int]]:
        music: List[Tuple[float, str, int]] = []
        sample_rate = float(self.hp.Sound.Sample_Rate)
        frame_shift = float(self.hp.Sound.Frame_Shift)
        ratio = sample_rate / frame_shift

        # load label
        with open(label_path) as f:
            while True:
                line = f.readline()

                if line == '' or not line:
                    break

                [start_time, end_time, lyric, note] = line.split(',')

                start_time = float(start_time)
                end_time = float(end_time)
                note = int(note)

                # match to the HiFiSinger implementation
                if lyric == 'pau' or lyric == 'sil' or lyric == 'xx':
                    lyric = '<X>'

                lyric = self.tokens[lyric]

                duration = end_time - start_time
                duration = int(duration * ratio)

                music.append((duration, lyric, note))

        return music

    def __plot_result(
            self,
            predicted_mels: torch.Tensor,
            predicted_silences: torch.Tensor,
            predicted_pitches: torch.Tensor,
            predicted_durations: torch.Tensor,
            png_path: str) -> None:
        for mel, silence, pitch, duration in zip(
            predicted_mels.cpu(),
            predicted_silences.cpu(),
            predicted_pitches.cpu(),
            predicted_durations.cpu(),
        ):
            title = 'Note infomation: {}'.format(png_path)
            new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0))
            plt.plot(silence)
            plt.margins(x=0)
            plt.title('Silence    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (2, 0))
            plt.plot(pitch)
            plt.margins(x=0)
            plt.title('Pitch    {}'.format(title))
            plt.colorbar()
            duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
            duration = torch.arange(duration.size(
                0)).repeat_interleave(duration)
            plt.subplot2grid((4, 1), (3, 0))
            plt.plot(duration)
            plt.margins(x=0)
            plt.title('Duration    {}'.format(title))
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close(new_Figure)


if __name__ == "__main__":
    print('inference start')
    inference = Inference(
        kInferenceAcousticModelCheckpointPath,
        kInferenceVocoderModelPath,
        kInferenceHyperParametersPath,
        kInferenceTokenDictPath,
    )
    inference(
        kInferenceInputLabelPath,
        kInferenceOutputWavefilePath,
        kInferenceOutputPlotPath,
    )
    print('inference done')
