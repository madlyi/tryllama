import whisper

class STT:
    def __init__(self):
        self.model = whisper.load_model("medium")

    def transcribe_audio(self, file_path):
        result = self.model.transcribe(file_path)
        return result['text'], result['language']
