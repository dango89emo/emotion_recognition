import os
import glob
from modules.mel_spectrogram import MelSpectrogram
from modules.constants import Path


def prepare_data_from_ravdess():
    """
    メルスペクトログラムの画像を生成する。
    使用する音声データは、KaggleのRAVDESS
    https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
    このデータは、声優が音声を感情を込めて吹き込んだもの。
    ファイル名に感情ラベルが含まれている。
    """
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

if __name__=="__main__":
    prepare_data_from_ravdess()