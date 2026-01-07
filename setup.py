from setuptools import setup, find_packages

setup(
    name="islamic-music-bot",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'audiocraft',
    ],
    description="Bot penghasil musik Islami original",
)
