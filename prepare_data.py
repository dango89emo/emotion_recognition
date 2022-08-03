import os
import glob
import boto3
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from modules.mel_spectrogram import MelSpectrogram
from modules.constants import Path, AWSConfig, TorchParams

"""
メルスペクトログラムの画像を生成する。
使用する音声データは、KaggleのRAVDESS
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
このデータは、声優が音声を感情を込めて吹き込んだもの。
ファイル名に感情ラベルが含まれている。

このスクリプトは、ローカルPCにより実行して、
音声データをメルスペクトログラムに加工し、
pytorchの学習データにしたのち、
AWS S3へと学習データをアップロードする。

前準備①：
　これを実行する前に、上記のページからデータをダウンロードし、
　プロジェクトルートに、audio-datasetの名前で解凍。
　その中にある、audio_speech_actors_01-24を削除
前準備②：
 　modules.constants.AWSConfigのbucketを、希望するバケット名に変更
前準備③：
   ローカルPCにて、AWS認証情報を準備して、上記バケットへのアクセス情報を用意。
"""

def ravdess2melspectrograms():
    mel = MelSpectrogram()

    # 感情ラベルと感情名の対応
    num2emotion={'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}

    counts = {}
    for actor in glob.glob(Path.AUDIO_FOLDER): 
        for path_to_audio_file in glob.glob(actor +'/*'):
            # ファイル名の例：　"03-01-02-02-01-01-01.wav"
            # 感情ラベルは、三つ目の数値(02)に含まれている。
            audio_file = path_to_audio_file.split("/")[-1]
            emotion_num=audio_file.split('-')[2]
            emotion=num2emotion[emotion_num]

            save_path_train = Path.OUTPUT_FOLDER_TRAIN + emotion
            save_path_test = Path.OUTPUT_FOLDER_TEST + emotion

            # 8つに1つは、検証用データとして活用する。
            count = counts.get(emotion, 1)
            print(f"emotion {emotion}, count {str(count)}")
            if (count % 8 == 0):
                picture_file = os.path.join(save_path_test, "{}{}.jpg".format(emotion, str(count).zfill(6)))
                os.makedirs(save_path_test, exist_ok=True)
            else:
                picture_file = os.path.join(save_path_train, "{}{}.jpg".format(emotion, str(count).zfill(6)))
                os.makedirs(save_path_train, exist_ok=True)   
            counts[emotion] = count + 1

            mel.make(path_to_audio_file, picture_file)

def melspectrograms2tensors():
    transform = transforms.Compose(
        [
            transforms.Resize(TorchParams.img_size),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    train_dataset = ImageFolder(root=Path.OUTPUT_FOLDER_TRAIN, transform=transform)
    test_dataset = ImageFolder(root=Path.OUTPUT_FOLDER_TEST , transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=TorchParams.hyper_param['batch-size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TorchParams.hyper_param['batch-size'], shuffle=True)

    torch.save(train_dataloader, Path.TRAIN_LOADER)
    torch.save(test_dataloader, Path.TEST_LOADER)


def upload_to_s3():
    s3_client = boto3.client('s3',region_name = AWSConfig.region)
    for filepath in glob.glob("./data/*.pt"):
        s3_client.upload_file(filepath, AWSConfig.bucket, filepath.replace("./data/", ""))


if __name__=="__main__":
    # ravdess2melspectrograms()
    melspectrograms2tensors()
    upload_to_s3()