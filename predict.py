import torch
from torchvision import PILToTensor
from modules.constants import model_name, classes
from modules.mel_spectrogram import MelSpectrogram
from PIL import Image

path_to_image="data/image.jpg"
path_to_wav="data/sound.wav"

mel = MelSpectrogram()
mel.make(path_to_wav, path_to_image)
img = PILToTensor(Image.open(path_to_image))

v = torch.load(model_name)

with torch.no_grad:
    preds = v(img)
    predicted_emotion = classes[preds[0].argmax(0)]

    print(f"この声の感情は、{predicted_emotion}です。")
