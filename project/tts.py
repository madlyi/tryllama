import torch
from TTS.api import TTS

class TTSHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def text_to_speech(self, text, lang, voice_to_mimic, output_file):
        self.tts.tts_to_file(text=text, speaker_wav=voice_to_mimic, language=lang, file_path=output_file)
