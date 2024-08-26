from openai import OpenAI
import os

client = OpenAI(api_key="")

# Transcribe the audio
audio_file = open("resources/5 wild new AI tools you can try right now - Fireship (youtube).mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

print(transcription)

# Define output directory and create it if it doesn't exist
output_dir = 'resources/transcriptions'
os.makedirs(output_dir, exist_ok=True)

# Construct the full path for the output file
output_file_path = os.path.join(output_dir, "transcription_output.txt")

# Export the transcription text to a .txt file
with open(output_file_path, "w") as text_file:
    text_file.write(transcription)

print(f"Transcription saved to {output_file_path}")