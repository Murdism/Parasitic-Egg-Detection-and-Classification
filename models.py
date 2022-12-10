
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_pretrained_model():
    
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    # remove the last layer
    resnet152_model = torch.nn.Sequential(*(list(resnet152.children())[:-1]))
    # freeze the model
    for param in resnet152_model.parameters():
        param.requires_grad = False

    return resnet152_model



class CNN(nn.Module):
    def __init__(self, input_channels=3, filters=32, num_classes=11):
        super(CNN, self).__init__()
        self.conv1  = nn.Conv2d(input_channels, filters, kernel_size=2) 
        self.conv2  = nn.Conv2d(filters, filters * 2, kernel_size=3)

        self.fc1  =  nn.Linear(in_features=  (filters * 2 * 56*56  + 2048) , out_features=filters)
        self.out = nn.Linear(filters, out_features=num_classes)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x, x2):
        x  = self.conv1(x) # output = (B, C,224,224)
        x  = F.relu(x) 
        x  = F.max_pool2d(x, kernel_size=2, padding=1) #output = (B,C,112,112)
        x  = F.dropout(x, 0.2)

        x  = self.conv2(x) #output = (B,iC,112,112)
        x  = F.relu(x)
        x  = F.max_pool2d(x, kernel_size=2, padding=1) # #output = (B,iC,56,56)
        x  = F.dropout(x, 0.2)
    
        x  = self.flatten(x)
        x2  = self.flatten(x2)
        x  = torch.cat((x, x2), dim=1)
        x  = self.fc1(x)
        x  = F.relu(x)
        x  = F.dropout(x, 0.25)
        x  = self.out(x)
        return x



class tinyCNN(nn.Module):
    def __init__(self, input_channels=3, filters=32, num_classes=11):
        super(tinyCNN, self).__init__()
        self.conv1  = nn.Conv2d(input_channels, filters, kernel_size=2) 
        self.conv2  = nn.Conv2d(filters, filters * 2, kernel_size=3)
        self.out = nn.Linear(filters * 2 * 56*56, out_features=num_classes)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x  = self.conv1(x) # output = (B, C,224,224)
        x  = F.relu(x) 
        x  = F.max_pool2d(x, kernel_size=2, padding=1) #output = (B,C,112,112)
        x  = F.dropout(x, 0.2)

        x  = self.conv2(x) #output = (B,iC,112,112)
        x  = F.relu(x)
        x  = F.max_pool2d(x, kernel_size=2, padding=1) # #output = (B,iC,56,56)
        x  = F.dropout(x, 0.2)

        x  = self.flatten(x)
        x  = self.out(x)
        return x
