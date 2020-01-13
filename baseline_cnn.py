# coding: utf-8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from tqdm import tqdm


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = sorted([os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)])
        self.crop_size = crop_size
        self.dict = {'拖网':0, '围网':1, '刺网':2}

    def __getitem__(self, index):
        name = self.image_filenames[index]
        target = name.split('_')[-1].split('.')[0]
        target = self.dict[target]
        img = Image.open(name)
        # 数据增强必须做吗？
        transform = transforms.Compose([transforms.Resize(size=(80, 60)),
                                        transforms.RandomCrop(size=(self.crop_size)), transforms.RandomRotation(degrees=90),
                                        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        tensor = transform(img)
        return tensor, target
    def __len__(self):
        return len(self.image_filenames)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc0 = nn.Linear(16384, 9216)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, val_loader, optimizer, epoch):
    model.train()
    train_bar = tqdm(enumerate(train_loader))
    val_bar = tqdm(val_loader)
    for batch_idx, (data, target) in train_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss/len(val_loader.dataset), correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--crop_size', type=int, default=32,
                        help='crop size for training (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_epoch', type=int, default=20,
                        help='Saving number')
    args = parser.parse_args()
    EPOCH_SAVE = args.save_epoch
    CROP_SIZE = args.crop_size
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_set = DatasetFromFolder('./point/train', crop_size=CROP_SIZE)
    train_size = int(0.8 * len(data_set))
    val_size = len(data_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.test_batch_size, shuffle=True, num_workers=1, pin_memory=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, val_loader, optimizer, epoch)
        scheduler.step()

        if epoch % EPOCH_SAVE == 0 and epoch != EPOCH_SAVE:
            torch.save(model.state_dict(), "baseline_cnn.pth")


if __name__ == '__main__':
    main()


# 裁剪是否有必要？
#