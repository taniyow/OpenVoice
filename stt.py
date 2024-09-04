from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI client with your API key
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Transcribe the audio
audio_file_path = "resources/5 wild new AI tools you can try right now - Fireship (youtube).mp3"
with open(audio_file_path, "rb") as audio_file:
    transcription = openai.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
  )

# Define output directory and create it if it doesn't exist
output_dir = 'resources/texts'
os.makedirs(output_dir, exist_ok=True)

# Function to translate text and save to a file
def translate_and_save(text, language_code, output_dir):
    system_prompt = "You are multilingual expert. Translate the following text to English, Spanish and French."
    user_prompt = f"Translate the following text to {language_code}:\n\n{text}"
    response = openai.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ]
    )
    translated_text = response.choices[0].message.content
    print(f"Translation to {language_code}:\n{translated_text}")
    output_file_path = os.path.join(output_dir, f"{language_code}.txt")
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(translated_text)
    print(f"Translation saved to {output_file_path}")

# Save the transcription
translate_and_save(transcription, "english", output_dir)  # English
# translate_and_save(transcription, "spanish", output_dir)  # Spanish
# translate_and_save(transcription, "french", output_dir)  # French
