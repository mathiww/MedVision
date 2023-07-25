from torch import nn, load
from torchvision import transforms

import numpy as np
from PIL import Image
from io import BytesIO


class LiverClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.class_names = []

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomPerspective(distortion_scale=.1, p=.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1380], std=[0.1735])
        ])

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=14*14*512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2),

            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2),
            nn.Linear(256, 17)
        )


    def load_weights(self, path="./app/models/LiverClassifier/best.pth.tar"):
        self.load_state_dict(load(path, map_location ='cpu'))
        

    def transforms(self, x):
        x = Image.open(BytesIO(x)).convert('L')

        return self.transform(x).unsqueeze(0)
    
    def forward(self, x):
        x = self.transforms(x)
        x = self.conv_layer(x)

        return self.dense_layer(x)
    

if __name__ == "__main__":
    net = LiverClassifier()
    print(net.requires_grad_())