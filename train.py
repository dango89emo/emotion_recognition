from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from torch import nn
from vit_pytorch import ViT

from modules.constants import Path, model_name, img_size

# ハイパーパラメーター
learning_rate = 1e-3
batch_size = 64
epochs = 20


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader): 
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=="__main__":
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    train_dataset = ImageFolder(root=Path.OUTPUT_FOLDER_TRAIN, transform=transform)
    test_dataset = ImageFolder(root=Path.OUTPUT_FOLDER_TEST , transform=transform)
    print(train_dataset.class_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    v = ViT(
        image_size = img_size * 2,
        patch_size = 32,
        num_classes = 8,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(v.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, v, loss_fn, optimizer)
        test_loop(test_dataloader, v, loss_fn)
    print("Learning Done!")

    torch.save(v, model_name)