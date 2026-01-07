from setuptools import setup, find_packages

setup(
    name="islamic-music-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'audiocraft',
        'setuptools'
    ],
    author="Bot",
    description="Automated Original Islamic Music Generator",
)
