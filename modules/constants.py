class Path:
    AUDIO_FOLDER = "audio-dataset/*"
    OUTPUT_FOLDER_TRAIN = "output_folder_train/"
    OUTPUT_FOLDER_TEST = "output_folder_test/"
    TRAIN_LOADER = "./data/train_dataloader.pt"
    TEST_LOADER = "./data/test_dataloader.pt"

class TorchParams:
    pretrained_model="vit_base_patch16_224_in21k"
    model_name = "audio_emotion_vit.pth"
    device="cuda"
    device2="cpu"
    img_size = (224,224)
    hyper_param = {
        'lr':3e-5,
        'batch-size':64,
        'epochs':50,
        'momentum':0.7   # for Adam
    }
    seed=42
    classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

class AWSConfig:
    bucket="sagemaker-studio-pxb5r0q2mbq" # 自分のバケット名を入力
    region="ap-northeast-1"
