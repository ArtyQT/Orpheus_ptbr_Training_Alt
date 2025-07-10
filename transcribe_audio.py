# -*- coding: utf-8 -*-
import whisper
import pyaudio
import wave
import os
import argparse
import tempfile
import sys

# Configuration constants for audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono)
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer

def record_audio(duration, output_filename="output.wav", sample_rate=RATE, channels=CHANNELS, audio_format=FORMAT, chunk_size=CHUNK):
    """Records audio from the microphone for a specified duration."""
    print(f"Recording for {duration} seconds...")
    audio = pyaudio.PyAudio()

    try:
        # Find the default input device
        input_device_index = audio.get_default_input_device_info()['index']
        print(f"Using input device: {audio.get_default_input_device_info()['name']}")

        # Open a new audio stream
        stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=chunk_size
        )

        frames = []

        # Read data in chunks
        for _ in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)

        print("Recording finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Terminate the PortAudio interface
        audio.terminate()

        # Save the recorded data as a WAV file
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved to {output_filename}")
        return output_filename

    except OSError as e:
        print(f"Error recording audio: {e}", file=sys.stderr)
        print("Please ensure you have a microphone connected and PyAudio is installed correctly.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}", file=sys.stderr)
        return None

def transcribe_audio_with_whisper(audio_filepath, model_size="base"):
    """Transcribes the audio file using the Whisper model."""
    print(f"Loading Whisper model '{model_size}'...")
    try:
        model = whisper.load_model(model_size)
        print("Whisper model loaded.")

        print(f"Transcribing audio from {audio_filepath}...")
        # Use transcribe with verbose=False to suppress progress bar
        result = model.transcribe(audio_filepath, verbose=False)
        print("Transcription complete.")
        return result
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        print("Please ensure the Whisper model files are downloaded correctly and you have enough memory.", file=sys.stderr)
        return None

def save_transcription_to_file(transcription_result, output_txt_filepath):
    """Saves the transcribed text to a file, one sentence per line."""
    if transcription_result is None or 'segments' not in transcription_result:
        print("No transcription segments found to save.", file=sys.stderr)
        return False

    try:
        with open(output_txt_filepath, 'w', encoding='utf-8') as f:
            # Whisper's segments are often sentence-like units
            for segment in transcription_result['segments']:
                text = segment['text'].strip()
                if text:
                    f.write(text + '\n')

        print(f"Transcription saved to {output_txt_filepath}")
        return True

    except Exception as e:
        print(f"Error saving transcription to file: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio and transcribe it using Whisper.")
    parser.add_argument("--duration", type=int, default=10, help="Duration of recording in seconds.")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to the output text file for transcription.")
    parser.add_argument("--whisper_model", type=str, default="base", choices=whisper.available_models(), help="Whisper model size to use.")

    args = parser.parse_args()

    # Create a temporary file for the audio recording
    # Use suffix='.wav' to ensure it's treated as a WAV file
    # Use delete=False so the file isn't deleted automatically when closed
    temp_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            temp_audio_filepath = tmpfile.name
        print(f"Using temporary audio file: {temp_audio_filepath}")

        # 1. Record audio
        recorded_file = record_audio(args.duration, output_filename=temp_audio_filepath)

        if recorded_file:
            # 2. Transcribe audio
            transcription = transcribe_audio_with_whisper(recorded_file, model_size=args.whisper_model)

            if transcription:
                # 3. Save transcription
                save_transcription_to_file(transcription, args.output_txt)

    finally:
        # Clean up the temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            print(f"Cleaned up temporary audio file: {temp_audio_file}")
        # Also check and remove the file created by tempfile.NamedTemporaryFile(delete=False)
        if 'temp_audio_filepath' in locals() and os.path.exists(temp_audio_filepath):
            os.remove(temp_audio_filepath)
            print(f"Cleaned up temporary audio file: {temp_audio_filepath}")

```