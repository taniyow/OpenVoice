import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

##############################################################

# Initialize OpenVoiceV2 using checkpoints_v2
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

##############################################################

# Obtain reference speaker or the voice you want to clone
# Then obtain the tone color embedding
reference_speaker = 'resources/5 wild new AI tools you can try right now - Fireship (youtube).mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

##############################################################

# Use MeloTTS as base speaker model
from melo.api import TTS

# Path to txt files
text_files = {
    'EN_NEWEST': "resources/texts/en.txt",  # The newest English base speaker model
    # 'EN': "resources/texts/en.txt",  # English
    # 'ES': "resources/texts/es.txt",  # Spanish
    # 'FR': "resources/texts/fr.txt",  # French
}

# Function to read the text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Read the texts from the files
texts = {language: read_text_from_file(file_path) for language, file_path in text_files.items()}

src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)