import os
import datetime
import sys

# Memaksa Python melihat folder audiocraft lokal hasil git clone
sys.path.insert(0, os.path.abspath("."))

def generate_bot_music():
    try:
        import torch
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        
        print("--- Memulai Bot Musik Islami ---")
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=30) 

        prompt = "Spiritual Islamic background music, acoustic oud, meditative ney flute, high quality"

        print("Sedang memproses AI (30 detik)...")
        wav = model.generate([prompt], progress=True)

        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        for one_wav in wav:
            # Fungsi ini akan menggunakan 'av' dari conda secara otomatis
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"BERHASIL! Cek folder '{output_dir}'")

    except Exception as e:
        print(f"TERJADI ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
