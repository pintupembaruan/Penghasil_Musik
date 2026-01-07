import os
import sys
import datetime

# Pastikan Python mengutamakan folder lokal hasil clone tadi
current_dir = os.getcwd()
sys.path.insert(0, os.path.join(current_dir, "audiocraft"))
sys.path.insert(0, current_dir)

def generate_bot_music():
    try:
        import torch
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        
        print("--- Memulai Sesi Produksi Musik Islami ---")
        
        # Menggunakan model 'small' untuk efisiensi CPU GitHub
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=30) 

        # Prompt musik Islami yang damai
        prompt = "Spiritual Islamic background music, acoustic oud, meditative ney flute, high quality"

        print("Sedang memproses melodi (estimasi 3-5 menit)...")
        # progress=True akan menampilkan bar proses di log GitHub
        wav = model.generate([prompt], progress=True)

        output_dir = 'music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{output_dir}/islamic_track_{timestamp}'

        print(f"Menyimpan file ke {filepath}.mp3...")
        for one_wav in wav:
            # audio_write akan otomatis menggunakan ffmpeg/av dari Conda
            audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
        
        print(f"BERHASIL! Cek folder 'music' setelah proses selesai.")

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    generate_bot_music()
