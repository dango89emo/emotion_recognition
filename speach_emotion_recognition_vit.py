import os
import numpy as np
import matplotlib as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import random
import timm
from tqdm.notebook import tqdm
from modules.constants import TorchParams, AWSConfig, Path
import seaborn as sns
import argparse
import json


def train_main(model, criterion, train_loader, valid_loader):
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(TorchParams.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(TorchParams.device)
            label = label.to(TorchParams.device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)              

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(TorchParams.device)
                label = label.to(TorchParams.device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        train_acc_list.append(epoch_accuracy)
        val_acc_list.append(epoch_val_accuracy)
        train_loss_list.append(epoch_loss)
        val_loss_list.append(epoch_val_loss)

    #出力したテンソルのデバイスをCPUへ切り替える
    device2 = torch.device(TorchParams.device2)

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    for i in range(TorchParams.epochs):
        train_acc2 = train_acc_list[i].to(device2)
        train_acc3 = train_acc2.clone().numpy()
        train_acc.append(train_acc3)

        train_loss2 = train_loss_list[i].to(device2)
        train_loss3 = train_loss2.clone().detach().numpy()
        train_loss.append(train_loss3)

        val_acc2 = val_acc_list[i].to(device2)
        val_acc3 = val_acc2.clone().numpy()
        val_acc.append(val_acc3)

        val_loss2 = val_loss_list[i].to(device2)
        val_loss3 = val_loss2.clone().numpy()
        val_loss.append(val_loss3)

    #取得したデータをグラフ化する
    sns.set()
    num_epochs = TorchParams.epochs

    plt.subplots(figsize=(12, 4), dpi=80)

    ax1 = plt.subplot(1,2,1)
    ax1.plot(range(num_epochs), train_acc, c='b', label='train acc')
    ax1.plot(range(num_epochs), val_acc, c='r', label='val acc')
    ax1.set_xlabel('epoch', fontsize='12')
    ax1.set_ylabel('accuracy', fontsize='12')
    ax1.set_title('training and val acc', fontsize='14')
    ax1.legend(fontsize='12')

    ax2 = plt.subplot(1,2,2)
    ax2.plot(range(num_epochs), train_loss, c='b', label='train loss')
    ax2.plot(range(num_epochs), val_loss, c='r', label='val loss')
    ax2.set_xlabel('epoch', fontsize='12')
    ax2.set_ylabel('loss', fontsize='12')
    ax2.set_title('training and val loss', fontsize='14')
    ax2.legend(fontsize='12')
    plt.show()


def seed_everything():
    random.seed(TorchParams.seed)
    os.environ['PYTHONHASHSEED'] = str(TorchParams.seed)
    np.random.seed(TorchParams.seed)
    torch.manual_seed(TorchParams.seed)
    torch.cuda.manual_seed(TorchParams.seed)
    torch.cuda.manual_seed_all(TorchParams.seed)
    torch.backends.cudnn.deterministic = True

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    return parser.parse_known_args()


def train():
    args, unknown = _parse_args()

    train_loader = torch.load(os.path.join(args.train, Path.TRAIN_LOADER.split("/")[-1]))
    test_loader = torch.load(os.path.join(args.train, Path.TEST_LOADER.split("/")[-1]))

    model=timm.create_model(TorchParams.pretrained_model, pretrained=True, num_classes=len(TorchParams.classes))
    model.to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TorchParams.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=TorchParams.gamma)
    seed_everything()
    train(model, criterion, train_loader, valid_loader)
    
    if args.current_host == args.hosts[0]:
        torch.save(model, os.path.join(args.sm_model_dir, TorchParams.model_name))

if __name__=="__main__":
    train()
