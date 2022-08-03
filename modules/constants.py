class Path:
    AUDIO_FOLDER = "audio-dataset/*"
    OUTPUT_FOLDER_TRAIN = "output_folder_train/"
    OUTPUT_FOLDER_TEST = "output_folder_test/"

model_name = "audio_emotion_vit.pth"
img_size = 256

classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

class AWSConfig:
    bucket="" # 自分のバケット名を入力
    region="ap-northeast-1"
