from stt import STT
from tts import TTSHandler
from query_handler import QueryHandler
from audio_recording import AudioRecorder

def main():
    stt = STT()
    tts_handler = TTSHandler()
    query_handler = QueryHandler()
    audio_recorder = AudioRecorder()

    # Record audio
    recorded_file_path = audio_recorder.live_recording()

    # Transcribe audio and get answer
    answer, recording_language = query_handler.transcribe_and_query(recorded_file_path, stt)

    # Convert answer to speech
    tts_handler.text_to_speech(answer, recording_language, "model_voice.mp3", "output.ogg")

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
