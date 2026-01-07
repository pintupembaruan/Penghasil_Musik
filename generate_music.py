import os
import datetime
import sys

def generate_bot_music():
    try:
        import torch
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        
        print("Memulai AI Model (MusicGen Small)...")
        
        # Load model ke CPU
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=60) # 1 menit

        prompt = "Calm spiritual Islamic background music, oud, ney, ambient, high quality"

        print("Sedang menghasilkan audio... Mohon tunggu.")
        wav = model.generate([prompt], progress=True)

        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        for one_wav in wav:
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"BERHASIL! Cek folder '{output_dir}' setelah action ini selesai.")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
