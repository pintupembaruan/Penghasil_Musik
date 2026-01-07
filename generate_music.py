import torch
import os
import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_bot_music():
    print("Memulai proses pembuatan musik Islami (Durasi: 1 Menit)...")
    
    # Menggunakan CPU karena GitHub Actions tidak punya GPU (tetap jalan namun agak lambat)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=60) # Set ke 60 detik (1 menit)
    
    # Prompt yang dioptimalkan untuk musik original tanpa CR
    prompt = "Peaceful Islamic background music, acoustic oud, ambient ney flute, slow tempo, high quality, no vocals, spiritual"

    # Proses generate
    wav = model.generate([prompt], progress=True)

    # Folder tujuan
    output_dir = 'music'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/islamic_track_{timestamp}'

    # Simpan sebagai mp3
    for one_wav in wav:
        # Kita simpan hasil akhir
        audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness", format="mp3")
    
    print(f"Selesai! File tersimpan di: {filename}.mp3")

if __name__ == "__main__":
    generate_bot_music()
