import torch
import os
import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_bot_music():
    print("Memproses musik... Mohon tunggu, ini memakan waktu sekitar 5-10 menit di CPU.")
    
    # Memaksa penggunaan CPU agar stabil di GitHub Actions
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=60) # 60 detik

    # Deskripsi musik agar original & islami
    prompt = "Calm Islamic spiritual ambient, oud instrument, soft percussion, peaceful, high quality"

    # Proses generate
    wav = model.generate([prompt], progress=True)

    output_dir = 'music'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f'{output_dir}/islamic_track_{timestamp}'

    # Simpan ke mp3
    for one_wav in wav:
        audio_write(filepath, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
    
    print(f"Selesai! File tersimpan sebagai {filepath}.mp3")

if __name__ == "__main__":
    generate_bot_music()
