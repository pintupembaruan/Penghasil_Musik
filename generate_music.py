import os
import datetime
import sys

def generate_bot_music():
    try:
        import torch
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        
        print("Memuat Model AI...")
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=30) 

        prompt = "Deep spiritual Islamic ney flute and oud, peaceful atmosphere, high quality"

        print("Sedang membuat musik (30 detik)...")
        wav = model.generate([prompt], progress=True)

        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        for one_wav in wav:
            # Di sini library 'av' akan digunakan untuk menyimpan MP3
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"BERHASIL: File tersimpan di {filepath}.mp3")

    except Exception as e:
        print(f"TERJADI ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
