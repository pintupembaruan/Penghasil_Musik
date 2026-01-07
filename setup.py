from setuptools import setup, find_packages

setup(
    name="islamic-music-bot",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        # Dependensi utama sudah dihandle di workflow yml 
        # untuk menghindari error build wheel
    ],
)
