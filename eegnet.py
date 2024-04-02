import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

from torch.utils.data import Dataset, DataLoader

class ShallowEEG(nn.Module):
    def __init__(self, in_chans, in_length, no_classes):
        super(ShallowEEG, self).__init__()
        dummy = torch.zeros((1,1,in_chans, in_length))
        
        self.convtemp = nn.Conv2d(1, 40, (1, 25))
        init.xavier_uniform_(self.convtemp.weight)
        init.constant_(self.convtemp.bias, 0)
        
        self.convspat = nn.Conv2d(40, 40, (in_chans, 1))
        init.xavier_uniform_(self.convspat.weight)
        init.constant_(self.convspat.bias, 0)
        
        self.pool = nn.MaxPool2d((1, 75), stride=(1, 15))
        dummy = nn.Flatten()(self.pool(self.convspat(self.convtemp(dummy))))
        self.linear = nn.Linear(dummy.shape[-1], no_classes)
    
    def forward(self, data):
        data = f.relu(self.convtemp(data))
        data = f.relu(self.convspat(data))
        data = self.pool(data)
        data = nn.Flatten()(data)
        data = f.softmax(self.linear(data))
        return data
    
class DeepEEG(nn.Module):
    def __init__(self, in_chans, in_length, no_classes):
        super(DeepEEG, self).__init__()
        dummy = torch.zeros((1,1,in_chans, in_length))
        # Block 1
        self.convtemp = nn.Conv2d(1,  25, (1, 10))
        self.wbinit(self.convtemp)
        
        self.convspat = nn.Conv2d(25, 25, (in_chans, 1))
        self.wbinit(self.convspat)
        
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        
        self.block1 = nn.Sequential(
            self.convtemp,
            self.convspat,
            self.pool1
        )
        dummy = self.block1(dummy)
        
        # Block 2
        self.conv2 = nn.Conv2d(25, 50, (1, 10))
        self.wbinit(self.conv2)
        
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        
        self.block2 = nn.Sequential(
            self.conv2,
            nn.ReLU(),
            self.pool2
        )
        dummy = self.block2(dummy)
        
        # Block 3
        self.conv3 = nn.Conv2d(50, 100, (1,10))
        self.wbinit(self.conv3)
        
        self.pool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        
        self.block3 = nn.Sequential(
            self.conv3,
            nn.ReLU(),
            self.pool3
        )
        dummy = self.block3(dummy)
        
        # Block 4
        self.conv4 = nn.Conv2d(100, 200, (1,10))
        self.wbinit(self.conv4)
        
        self.pool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        
        self.block4 = nn.Sequential(
            self.conv4,
            nn.ReLU(),
            self.pool4
        )
        dummy = self.block4(dummy)
        
        # CLassifier
        dummy = nn.Flatten()(dummy)
        self.linear = nn.Linear(dummy.shape[-1], no_classes)
        
    
    def wbinit(self, layer):
        init.xavier_uniform_(layer.weight)
        init.constant_(layer.bias, 0)
    
    def forward(self, data):
        data = self.block1(data)
        data = self.block2(data)
        data = self.block3(data)
        data = self.block4(data)
        data = nn.Flatten()(data)
        data = f.softmax(self.linear(data))
        return data
        
    
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx]).unsqueeze(dim=0)
        y = self.y[idx]
        return X, y