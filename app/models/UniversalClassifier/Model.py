from torch import nn, load
from torchvision import transforms

import numpy as np
from PIL import Image
from io import BytesIO
from math import floor

class UniversalClassifier(nn.Module):
    def __init__(self): # Defining features configuration
        super().__init__()

        shape = 64
        in_channels = 3
        out_channels = 32
        num_conv_layers = 3 
        final_output = out_channels*(2**num_conv_layers) # Max image dimension

        # Sorted by index
        self.class_names = ["Sangue", "MRI do Encéfalo", "CT do Peito", "RX do Peito", "MRI do Joelho", "RX do Joelho", "MRI do Fígado", "Ocular", "Inesperada"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([64, 64]),
            transforms.Normalize(mean=[0.3720, 0.3410, 0.4056], std=[0.2808, 0.2760, 0.3118]),
        ])

        self.features = nn.ModuleList() # List of convolutions layers

        for _ in range(num_conv_layers):
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(0.2)
                )
            )
            in_channels = out_channels
            out_channels = out_channels*2 if out_channels*2 < final_output else out_channels
            shape = (floor(shape)-2)/2 # Auto image resolution calculation


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(shape*shape)*out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 9),
        )


    def load_weights(self, path="./app/models/UniversalClassifier/best.pth.tar"):
        self.load_state_dict(load(path, map_location='cpu')["state_dict"])


    def transforms(self, x):
        x = np.array(Image.open(BytesIO(x)).convert("RGB"))
        
        return self.transform(x).unsqueeze(0)


    def forward(self, x):
        x = self.transforms(x)

        for conv in self.features:
            x = conv(x)

        return self.classifier(x)
    

if __name__ == "__main__":
    net = UniversalClassifier()
    print(net.requires_grad_())
        