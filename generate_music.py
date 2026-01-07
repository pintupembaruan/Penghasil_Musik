import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import datetime

def generate_bot_music():
    print("Memulai proses pembuatan musik Islami...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    
    # Set durasi per segmen (30 detik x 6 = 3 menit)
    model.set_generation_params(duration=30)
    
    # Prompt spesifik untuk musik Islami original
    prompt = "Authentic Islamic ambient, peaceful Oud and Nay flute, rhythmic Duff percussion, spiritual atmosphere, high quality, no vocals"

    # Generate audio
    wav = model.generate([prompt])

    # Pastikan folder music ada
    if not os.path.exists('music'):
        os.makedirs('music')

    # Nama file berdasarkan timestamp agar unik
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'music/islamic_track_{timestamp}'

    # Simpan hasil
    for one_wav in wav:
        audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
    
    print(f"Berhasil menyimpan: {filename}.mp3")

if __name__ == "__main__":
    generate_bot_music()
