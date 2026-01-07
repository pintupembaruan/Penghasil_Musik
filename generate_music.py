import os
import datetime
import sys

# Memastikan script bisa melihat folder audiocraft yang kita download manual
sys.path.append(os.path.abspath("."))

def generate_bot_music():
    try:
        import torch
        # Mengimpor dari folder lokal
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        
        print("Model AI Siap. Memulai proses...")
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=30) 

        prompt = "Deep spiritual Islamic ney flute, peaceful atmosphere, no vocals, high quality"

        print("Sedang memproses audio AI (30 detik)...")
        wav = model.generate([prompt], progress=True)

        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        for one_wav in wav:
            # Menggunakan biner 'av' yang diinstal lewat conda
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"BERHASIL! Cek folder '{output_dir}'")

    except Exception as e:
        print(f"TERJADI KESALAHAN: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
