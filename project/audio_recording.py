import sounddevice as sd
from scipy.io.wavfile import write

class AudioRecorder:
    def live_recording(self, duration=10, file_path='user_input.wav'):
        fs = 44100

        print("Recording...")
        # Record audio
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        # Save the recording
        write(file_path, fs, myrecording)
        print(f"Audio saved as {file_path}")

        return file_path
