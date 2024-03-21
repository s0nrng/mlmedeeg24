import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
import numpy as np
from sklearn.model_selection import train_test_split



class SEBlock(nn.Module):
    def __init__(self, input_size, no_chans):
        super(SEBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool1d(input_size),
            nn.Flatten(),
            nn.Linear(no_chans, 128),
            nn.ReLU(),
            nn.Linear(128, no_chans),
            nn.Unflatten(1, torch.Size([no_chans, 1])),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        identity = image
        output = self.layers(image)
        identity = identity * output.expand_as(identity)
        return identity
    
class Bottleneck(nn.Module):
    def __init__(self, in_chans, k, mode = 'UP'):
        super(Bottleneck, self).__init__()
        self.mode = mode
        if mode == 'UP':
            self.bnorm1 = nn.BatchNorm1d(in_chans*k)
            self.conv1 = nn.Conv1d(in_chans*k, in_chans//2, 1, padding='same')
            init.xavier_uniform_(self.conv1.weight)
            init.constant_(self.conv1.bias, 0)
    
            self.bnorm2 = nn.BatchNorm1d(in_chans//2)
            self.conv2 = nn.Conv1d(in_chans//2, in_chans//2, 3, padding='same')
            init.xavier_uniform_(self.conv2.weight)
            init.constant_(self.conv2.bias, 0)
            
            self.bnorm3 = nn.BatchNorm1d(in_chans//2)
            self.conv3 = nn.Conv1d(in_chans//2, in_chans, 1, padding='same')
            init.xavier_uniform_(self.conv3.weight)
            init.constant_(self.conv3.bias, 0)
            
            self.bnorms = nn.BatchNorm1d(in_chans*k)
            self.shortcut = nn.Conv1d(in_chans*k, in_chans, 1, 1, bias=False)
            
        else:
            self.bnorm1 = nn.BatchNorm1d(in_chans)
            self.conv1 = nn.Conv1d(in_chans, in_chans//4, 1, padding='same')
            init.xavier_uniform_(self.conv1.weight)
            init.constant_(self.conv1.bias, 0)

            self.bnorm2 = nn.BatchNorm1d(in_chans//4)
            self.conv2 = nn.Conv1d(in_chans//4, in_chans//4, 3, padding='same')
            init.xavier_uniform_(self.conv2.weight)
            init.constant_(self.conv2.bias, 0)
            
            self.bnorm3 = nn.BatchNorm1d(in_chans//4)
            self.conv3 = nn.Conv1d(in_chans//4, in_chans, 1, padding='same')
            init.xavier_uniform_(self.conv3.weight)
            init.constant_(self.conv3.bias, 0)
        
    def forward(self, x):
        mode = self.mode
        identity = x
        x = self.conv1(f.relu(self.bnorm1(x)))
        x = self.conv2(f.relu(self.bnorm2(x)))
        x = self.conv3(f.relu(self.bnorm3(x)))
        
        if mode == 'UP':
            identity = self.shortcut(f.relu(self.bnorms(identity)))
        else:
            pass
        output = x + identity
        return output
    
class URBlock(nn.Module):
    def __init__(self, in_chans, k):
        super(URBlock, self).__init__()
        self.in_chans = in_chans
        self.k = k
        self.bottleneck = Bottleneck(in_chans, k)
        
    def forward(self, bs, ts):
        k = self.k
        Bs = torch.cat((bs, ts), dim=1)
        r = Bs.view(Bs.shape[0], Bs.shape[1]//k, k, Bs.shape[2]).sum(2)
        M = self.bottleneck(Bs)
        output = M+r
        return output

class DRBlock(nn.Module):
    def __init__(self, in_chans, k):
        super(DRBlock, self).__init__()
        self.in_chans = in_chans
        self.bottleneck = Bottleneck(in_chans, k, mode='DOWN')
        
    def forward(self,tensor_list):
        Hs = torch.cat(tensor_list, dim=1)
        return Hs + self.bottleneck(Hs)

class FishTail(nn.Module):
    def __init__(self):
        super(FishTail, self).__init__()
        # 750 -> 320
        self.conv1 = nn.Conv1d(22, 64, 2, 2)
        init.xavier_uniform_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)
        self.bnorm1 = nn.BatchNorm1d(64)
        
        # 320 -> 160
        self.conv2 = nn.Conv1d(64, 128, 2, 2)
        init.xavier_uniform_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0)
        self.bnorm2 = nn.BatchNorm1d(128)
        
        # 160 -> 80
        self.conv3 = nn.Conv1d(128, 256, 2, 2)
        init.xavier_uniform_(self.conv3.weight)
        init.constant_(self.conv3.bias, 0)
        self.bnorm3 = nn.BatchNorm1d(256)
        
        # 80 -> 40
        self.conv4 = nn.Conv1d(256, 512, 2, 2)
        init.xavier_uniform_(self.conv4.weight)
        init.constant_(self.conv4.bias, 0)
        self.bnorm4 = nn.BatchNorm1d(512)
        
        self.seblock = SEBlock(input_size=(40), no_chans=512)

    def forward(self, image):
        t1 = self.bnorm1(self.conv1(image))
        
        t2 = f.relu(t1)
        t2 = self.bnorm2(self.conv2(t2))
        
        t3 = f.relu(t2)
        t3 = self.bnorm3(self.conv3(t3))
        
        t4 = f.relu(t3)
        t4 = self.bnorm4(self.conv4(t4))
        
        b4 = self.seblock(t4)
        return t1, t2, t3, t4, b4

class FishBody(nn.Module):
    def __init__(self):
        super(FishBody, self).__init__()
        self.up3 = nn.Sequential(
            nn.Conv1d(512, 256, 2, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.ur3 = URBlock(256, 2)
        self.bnorm3 = nn.BatchNorm1d(256)
        
        self.up2 = nn.Sequential(
            nn.Conv1d(256, 128, 2, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.ur2 = URBlock(128, 2)
        self.bnorm2 = nn.BatchNorm1d(128)
        
        self.up1 = nn.Sequential(
            nn.Conv1d(128, 64, 2, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.bnorm1 = nn.BatchNorm1d(64)
        self.ur1 = URBlock(64, 2)
        
    def forward(self, t1, t2, t3, t4, b4):
        b3 = f.interpolate(self.up3(b4), scale_factor=2)
        b3 = self.bnorm3(self.ur3(b3, t3))
        
        b2 = f.interpolate(self.up2(b3), scale_factor=2)
        b2 = self.bnorm2(self.ur2(b2, t2))
        
        b1 = f.interpolate(self.up1(b2), scale_factor=2)
        b1 = self.bnorm1(self.ur1(b1, t1))
        return b1, b2, b3

class FishHead(nn.Module):
    def __init__(self):
        super(FishHead, self).__init__()
        self.maxpool2 = nn.MaxPool1d(2)
        in_chans = 64 + 128*2
        self.dr2 = DRBlock(in_chans, k=None)
        self.bnorm2 = nn.BatchNorm1d(in_chans)
        self.maxpool3 = nn.MaxPool1d(2)
        in_chans += 256*2
        self.dr3 = DRBlock(in_chans, k=None)
        self.bnorm3 = nn.BatchNorm1d(in_chans)
        self.maxpool4 = nn.MaxPool1d(2)
        in_chans += 512*2
        self.dr4 = DRBlock(in_chans, None)
        self.bnorm4 = nn.BatchNorm1d(in_chans)
    def forward(self, t1, t2, t3, t4, b1, b2, b3, b4):
        h2 = self.maxpool2(b1)
        h2 = self.bnorm2(self.dr2((h2, b2, t2)))
        h3 = self.maxpool3(h2)
        h3 = self.bnorm3(self.dr3((h3, b3, t3)))
        h4 = self.maxpool4(h3)
        h4 = self.bnorm4(self.dr4((h4, b4, t4)))
        return h4

class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        self.tail = FishTail()
        self.body = FishBody()
        self.head = FishHead()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1856),
            nn.ReLU(),
            nn.Conv1d(1856, 928, 1, bias=True),
            nn.BatchNorm1d(928),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(928, 500, 1, bias=True),
            nn.Flatten(),
            nn.Linear(500, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        t1, t2, t3, t4, b4 = self.tail(x)
        b1, b2, b3 = self.body(t1, t2, t3, t4, b4)
        output = self.head(t1, t2, t3, t4, b1, b2, b3, b4)
        output = self.classifier(output)
        return output

class FishDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.data[idx])
        # X = torch.permute(torch.unsqueeze(X, dim=0), (0,2,1))
        y = self.label[idx]
        return X[:,55:695], y
    
def FishLoader(data, testSize = 0.2, batch_size = 32):
    if len(data) == 2:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize)
    else:
        [X_train, y_train, X_test, y_test] = data

    train_dataset = FishDataset(X_train, y_train)
    test_dataset = FishDataset(X_test, y_test)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory = True, batch_size= batch_size)

    return train_loader, test_loader