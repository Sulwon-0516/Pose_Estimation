import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

NUM_KEYS = 17

# Define the model
class baseline(nn.Module):
    def __init__(self):
        super(baseline,self).__init__()

        backbone = torchvision.models.resnet50(pretrained = True)
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))

        self.Deconv1 = nn.ConvTranspose2d(in_channels = 2048, out_channels = 256, kernel_size = 4, stride = 2, padding = [1,1])
        self.BN1 = nn.BatchNorm2d(num_features = 256)
        self.Deconv2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 4, stride = 2, padding = [1,1])
        self.BN2 = nn.BatchNorm2d(num_features = 256)
        self.Deconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 4, stride = 2, padding = [1,1])
        self.BN3 = nn.BatchNorm2d(num_features = 256)
        self.Conv1x1 = nn.Conv2d(in_channels = 256, out_channels = NUM_KEYS, kernel_size = 1)

    def forward(self, input):
        # input is 256 x 192 size image.
        # output should be 64 by 48 (since downsized 5 times, up-sized 3 times)
        out = self.backbone(input)
        out = self.Deconv1(out)
        out = F.relu(self.BN1(out))
        #print("(deconv1):",out.shape)
        out = self.Deconv2(out)
        out = F.relu(self.BN2(out))
        #print("(deconv2):",out.shape)
        out = self.Deconv3(out)
        out = F.relu(self.BN3(out))
        #print("(deconv3):",out.shape)
        out = self.Conv1x1(out)

        return out
