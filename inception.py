import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.model_selection import train_test_split
import numpy as np

class InceptionDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.data[idx])
        X = torch.permute(torch.unsqueeze(X, dim=0), (0,2,1))
        y = self.label[idx]
        return X, y

def InceptionLoader(data, testSize = 0.2, batch_size = 32):
    if len(data) == 2:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize)
    else:
        [X_train, y_train, X_test, y_test] = data

    train_dataset = InceptionDataset(X_train, y_train)
    test_dataset = InceptionDataset(X_test, y_test)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory = True, batch_size= batch_size)

    return train_loader, test_loader

class Inception1(nn.Module):
    def __init__(self,
                scales_samples,
                in_chans,
                filters_per_branch,
                input_time_length,
                dropout_rate):
        super(Inception1, self).__init__()
        x0 = torch.ones((1,1,input_time_length,in_chans), dtype = torch.float32)
        unit = x0
        self.conv = nn.Conv2d(in_channels=unit.shape[1],
                              out_channels=filters_per_branch,
                              kernel_size=(scales_samples, 1),
                              padding='same')
        init.xavier_uniform_(self.conv.weight, gain=1)
        init.constant_(self.conv.bias, 0)

        unit = self.conv(unit)
        self.bnorm1 = nn.BatchNorm2d(unit.shape[1])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.depthconv = nn.Conv2d(in_channels=unit.shape[1],
                             out_channels=2*unit.shape[1],
                             kernel_size=(1,in_chans),
                             )
        init.xavier_uniform_(self.depthconv.weight, gain=1)
        init.constant_(self.depthconv.bias, 0)
        unit=self.depthconv(unit)
        self.bnorm2 = nn.BatchNorm2d(unit.shape[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
    def forward(self, data):
        data = self.conv(data)
        data = self.bnorm1(data)
        data = self.relu1(data)
        data = self.drop1(data)
        data = self.depthconv(data)
        data = self.bnorm2(data)
        data = self.relu2(data)
        data = self.drop2(data)
        return data
    
class Inception2(nn.Module):
    def __init__(self,
                b1_out,
                scales_samples,
                filters_per_branch,
                dropout_rate):
        super(Inception2, self).__init__()
        unit = b1_out
        self.conv = nn.Conv2d(in_channels=unit.shape[1],
                        out_channels=filters_per_branch,
                        kernel_size=(int(scales_samples/4), 1),
                        padding='same')
        init.xavier_uniform_(self.conv.weight, gain=1)
        init.constant_(self.conv.bias, 0)
        unit = self.conv(unit)

        self.bnorm = nn.BatchNorm2d(unit.shape[1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
    def forward(self, data):
        data = self.conv(data)
        data = self.bnorm(data)
        data = self.relu(data)
        data = self.drop(data)
        return data
    
class EEGInception(nn.Module):
    def __init__(self,
                in_chans=22,
                input_time_length=750,
                n_classes=3,
                filters_per_branch=8,
                scales_time=(500,250,125),
                dropout_rate=0.25):
        super(EEGInception, self).__init__()
        scales_samples = [int(s*input_time_length/1000) for s in  scales_time]
        
        #       ====== Inception Block 1 ======
        b1_units = []
        x0 = torch.ones((1,1,750,22), dtype = torch.float32)
        self.inception11 = Inception1(in_chans=in_chans,
                                      filters_per_branch=filters_per_branch,
                                      input_time_length=input_time_length,
                                      scales_samples=scales_samples[0],
                                      dropout_rate=dropout_rate
                                      )
        self.inception12 = Inception1(in_chans=in_chans,
                                      filters_per_branch=filters_per_branch,
                                      input_time_length=input_time_length,
                                      scales_samples=scales_samples[0],
                                      dropout_rate=dropout_rate
                                      )
        self.inception13 = Inception1(in_chans=in_chans,
                                      filters_per_branch=filters_per_branch,
                                      input_time_length=input_time_length,
                                      scales_samples=scales_samples[0],
                                      dropout_rate=dropout_rate
                                      )

        b1_units.append(self.inception11(x0))
        b1_units.append(self.inception12(x0))
        b1_units.append(self.inception13(x0))
        
        b1_out = torch.cat(b1_units, dim=1)
        self.avgpool1 = nn.AvgPool2d((4,1))
        b1_out = self.avgpool1(b1_out)
#       ====== Inception Block 2 ======
        b2_units = []
        self.inception21 = Inception2(scales_samples=scales_samples[0],
                                      b1_out=b1_out,
                                      filters_per_branch=filters_per_branch,
                                      dropout_rate=dropout_rate)
        self.inception22 = Inception2(scales_samples=scales_samples[1],
                                      b1_out=b1_out,
                                      filters_per_branch=filters_per_branch,
                                      dropout_rate=dropout_rate)
        self.inception23 = Inception2(scales_samples=scales_samples[2],
                                      b1_out=b1_out,
                                      filters_per_branch=filters_per_branch,
                                      dropout_rate=dropout_rate)
            
        b2_units.append(self.inception21(b1_out))            
        b2_units.append(self.inception22(b1_out))            
        b2_units.append(self.inception23(b1_out))
        
        b2_out = torch.cat(b2_units, dim=1)
        self.avgpool2 = nn.AvgPool2d((2,1))
        b2_out = self.avgpool2(b2_out)
#       ====== Output Block 3.1 ======
        b3_u1 = b2_out
    
        conv1 = nn.Conv2d(in_channels=b3_u1.shape[1],
                          out_channels=int(filters_per_branch*len(scales_samples)/2),
                          kernel_size=(8,1),
                          padding='same')
        init.xavier_uniform_(conv1.weight, gain=1)
        init.constant_(conv1.bias, 0)
        b3_u1 = conv1(b3_u1)

        bnorm1 = nn.BatchNorm2d(b3_u1.shape[1])
        relu1 = nn.ReLU()
        avgpoolb31 = nn.AvgPool2d((2,1))
        b3_u1 = avgpoolb31(b3_u1)
        drop1 = nn.Dropout()
        self.out1 = nn.Sequential(
            conv1,
            bnorm1,
            relu1,
            avgpoolb31,
            drop1
        )
#       ====== Output Block 3.2 ======
        b3_u2 = b3_u1
        conv2 = nn.Conv2d(in_channels=b3_u2.shape[1],
                          out_channels=int(filters_per_branch*len(scales_samples)/4),
                          kernel_size=(4,1),
                          padding='same')
        init.xavier_uniform_(conv2.weight, gain=1)
        init.constant_(conv2.bias, 0)
        b3_u2 = conv2(b3_u2)

        bnorm2 = nn.BatchNorm2d(b3_u2.shape[1])
        relu2 = nn.ReLU()
        avgpoolb32 = nn.AvgPool2d((2,1))
        b3_u2 = avgpoolb32(b3_u2)
        drop2 = nn.Dropout()
        self.out2 = nn.Sequential(
            conv2,
            bnorm2,
            relu2,
            avgpoolb32,
            drop2
        )
        b3_out = b3_u2
#       ====== Classify Block ======
        flatten = nn.Flatten()
        out = flatten(b3_out)
        linear = nn.Linear(in_features=out.shape[1],
                           out_features=n_classes)
        relu = nn.ReLU()
        out = linear(out)
        self.classify = nn.Sequential(
            flatten,
            linear,
            relu
        )
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, data):
        #Block1
        b1_units = list()
        b1_units.append(self.inception11(data))
        b1_units.append(self.inception12(data))
        b1_units.append(self.inception13(data))
        b1_out = torch.cat(b1_units, dim=1)
        b1_out = self.avgpool1(b1_out)
        #Block2
        b2_units = list()
        b2_units.append(self.inception21(b1_out))
        b2_units.append(self.inception22(b1_out))
        b2_units.append(self.inception23(b1_out))
        b2_out = torch.cat(b2_units, dim=1)
        b2_out = self.avgpool2(b2_out)
        #Out
        b3_u1 = self.out1(b2_out)
        b3_out = self.out2(b3_u1)
        out = self.classify(b3_out)
        out = self.softmax(out)
        return out