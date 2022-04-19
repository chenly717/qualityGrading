import  matplotlib.pyplot as plt
import numpy as np
import glob
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import shutil
from tensorboardX import SummaryWriter


data_dir = '/home/data'
image_path_list_train = glob.glob(os.path.join(data_dir, 'train', '*', '*'))
image_path_list_train.sort()
print(len(image_path_list_train))
image_path_list_val = glob.glob(os.path.join(data_dir, 'val', '*', '*'))
image_path_list_val.sort()
print(len(image_path_list_val))

categories = [d.name for d in os.scandir(os.path.join(data_dir, 'train')) if d.is_dir()]
categories.sort()
print(len(categories))
class2idx = {categories[i] : i for i in range(len(categories))}
idx2class = {idx : class_ for class_, idx in class2idx.items()}

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

class SceneDataset(Dataset):
    def __init__(self, samples, transform):
        self.transform = transform
        self.samples = samples
        image = []
        targets = []
        for path, target in self.samples:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                image.append(img)
            targets.append(target)
        self.image = image
        self.targets = targets
        print(len(self.samples))

    def get_item(self, index):
        img = self.image[index]
        target = self.targets[index]
        return img, target

    def __getitem__(self, index):
        sample, target = self.get_item(index)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

batch_size = 64
data = {
    'train':
         SceneDataset([(x, class2idx[x.split('/')[-2]]) for x in image_path_list_train], transform = image_transforms['train']),
     'val':
         SceneDataset([(x, class2idx[x.split('/')[-2]]) for x in image_path_list_val], transform =  image_transforms['val']),
         }

dataLoader = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size= batch_size, shuffle=False),
}

for step, (inputs, targets) in tqdm(enumerate(dataLoader['train'])):
    pass

model = torchvision.models.resnext101_32x8d(pretrained=True)
for i, layer in enumerate(model.children()):
    if i < 6:
        for param in layer.parameters():
            param.requires_grad = False

n_inputs = model.fc.in_features
model.fc = nn.Linear(n_inputs, 4)

def train_val(net, criterion, optimizer, train_loader, val_loader):
    _ = net.train() 
    train_loss = 0
    train_acc = 0
    for step, (inputs, targets) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, pred = torch.max(outputs, dim=1)
        correct_tensor = pred.eq(targets.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc += accuracy.item()

    train_loss, train_acc = train_loss / len(train_loader), train_acc / len(train_loader)

    _ = net.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _,pred = torch.max(outputs, dim=1)
            correct_tensor = pred.eq(targets.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            val_acc += accuracy.item()

    val_loss, val_acc = val_loss / len(val_loader), val_acc / len(val_loader)

    return train_loss, train_acc, val_loss, val_acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print('train batch:%s' % len(dataLoader['train']))
print('val batch:%s' % len(dataLoader['val']))

criterion = nn.CrossEntropyLoss()
net = model.cuda()
net = nn.DataParallel(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [10, 20, 30, 40], gamma=0.1)
version = 'bestAcc_para'
log_dir = '/home'
save_dir = os.path.join(log_dir, version)
if not os.path.join(save_dir):
    os.makedirs(save_dir)
else:
    shutil.rmtree(save_dir)
    os.makedirs(save_dir)

best_acc = 0
writer_train = SummaryWriter('/home/train') #数据存放在这个文件夹
writer_val = SummaryWriter('/home/val')
writer_train.add_scalar('loss', 10, 0)
writer_train.add_scalar('accuracy', 0, 0)
writer_val.add_scalar('loss', 10, 0)
writer_val.add_scalar('accuracy', 0, 0)

for epoch in range(48):
    scheduler.step()
    print('\n Version: %s Epoch: %d | learning rate:%f' % (version, epoch, get_lr(optimizer)))
    train_loss, train_acc, val_loss, val_acc = train_val(net, criterion, optimizer, dataLoader['train'], dataLoader['val'])
    writer_train.add_scalar('loss', train_loss, epoch+1)
    writer_train.add_scalar('accuracy', train_acc, epoch+1)
    writer_val.add_scalar('loss', val_loss, epoch+1)
    writer_val.add_scalar('accuracy', val_acc, epoch+1)
    print(epoch, train_loss, train_acc, val_loss, val_acc)

    if val_acc > best_acc:
        best_acc = val_loss
        save_path = os.path.join(save_dir, 'best_acc.pth')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'loss': val_loss,
            'acc': val_acc
        }
        torch.save(state, save_path)
