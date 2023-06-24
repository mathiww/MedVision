from torch import nn
from math import floor

class Model(nn.Module):
    def __init__(self, num_classes, size_image=64, in_channels=3, out_channels=32, num_conv_layers=3): # Defining features configuration
        super().__init__()

        _dim = size_image
        _in = in_channels
        _out = out_channels
        _final_output = _out*(2**num_conv_layers) # Max image dimension

        self.features = nn.ModuleList() # List of convolutions layers

        for _ in range(num_conv_layers):
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=0, bias=False),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(_out),
                    nn.ReLU(),
                    nn.Dropout2d(0.2)
                )
            )
            _in = _out
            _out = _out*2 if _out*2 < _final_output else _out
            _dim = (floor(_dim)-(in_channels-1))/2 # Auto image resolution calculation


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(_dim**2)*_out, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        for conv in self.features:
            x = conv(x)
        return self.classifier(x)
    
# if __name__ == "__main__":
#     net = UniversalClassifierModel()
#     print(net.requires_grad_())
        