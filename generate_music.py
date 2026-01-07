import torch
import os
import datetime
import sys
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_bot_music():
    try:
        print("Mulai proses AI... Ini membutuhkan waktu sekitar 5 menit.")
        
        # Inisialisasi model small
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=60) # Durasi 1 Menit

        # Prompt musik Islami Original
        prompt = "Calm and spiritual Islamic background music, acoustic oud, meditative ney flute, high quality, no vocals"

        # Proses pembuatan
        wav = model.generate([prompt], progress=True)

        # Pastikan folder ada
        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Penamaan file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        # Simpan ke MP3
        for one_wav in wav:
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"Sukses! File tersimpan di folder {output_dir}")

    except Exception as e:
        print(f"Terjadi error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
