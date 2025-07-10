#!/bin/bash

# Define output file paths and directories
TRANSCRIPTION_OUTPUT_TXT="./transcribed_input.txt"
AUDIO_OUTPUT_DIR="./generated_audio"

# --- Configuration ---
# Set your desired recording duration in seconds
RECORDING_DURATION=15
# Set the Whisper model size (e.g., base, small, medium, large)
WHISPER_MODEL="base"
# Set the path to your model checkpoint
CHECKPOINT_PATH="./outputs/checkpoint-latest" # <--- IMPORTANT: Update this path
# Set the base model name used during fine-tuning
BASE_MODEL="unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # <--- IMPORTANT: Update if different
# Set the device (cuda or cpu)
DEVICE="cuda"
# Set max new tokens for generation (optional, adjust as needed)
MAX_NEW_TOKENS=2000

# Ensure output directories exist
mkdir -p "$AUDIO_OUTPUT_DIR"

echo "--- Starting Audio Transcription ---"
# Run the transcription script
python transcribe_audio.py \
  --duration "$RECORDING_DURATION" \
  --output_txt "$TRANSCRIPTION_OUTPUT_TXT" \
  --whisper_model "$WHISPER_MODEL"

# Check if the transcription was successful (basic check: if the output file was created and is not empty)
if [ ! -s "$TRANSCRIPTION_OUTPUT_TXT" ]; then
  echo "Error: Transcription failed or resulted in an empty file. Aborting." >&2
  exit 1
fi

echo "--- Transcription Complete. Starting Audio Generation ---"
# Run the inference script using the transcribed text
python inference.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --base_model "$BASE_MODEL" \
  --input_txt "$TRANSCRIPTION_OUTPUT_TXT" \
  --output_dir "$AUDIO_OUTPUT_DIR" \
  --device "$DEVICE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

# Check if inference script ran successfully (check exit code)
if [ $? -eq 0 ]; then
  echo "---\nFull pipeline finished successfully.\n---"
  echo "Generated audio saved in: $AUDIO_OUTPUT_DIR"
else
  echo "---\nError during audio generation.\n---" >&2
  exit 1
fi

# Clean up the temporary transcription file (optional)
# rm "$TRANSCRIPTION_OUTPUT_TXT"
# echo "Cleaned up temporary transcription file: $TRANSCRIPTION_OUTPUT_TXT"
