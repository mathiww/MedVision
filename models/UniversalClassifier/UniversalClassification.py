from torch import load
from torchvision import transforms
from torch.nn.functional import softmax

from models.UniversalClassifier.UniversalClassifierModel import UniversalClassifierModel

from PIL import Image
import numpy as np
from io import BytesIO


# VARIABLES


TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([64, 64]),
    transforms.Normalize(mean=[0.3720, 0.3410, 0.4056],
                        std=[0.2808, 0.2760, 0.3118]),
])

CLASSES = ["Sangue", "MRI do Encéfalo", "CT do Peito", "RX do Peito", "MRI do Joelho", "RX do Joelho", "MRI do Fígado", "Ocular", "Inesperada"]

NET = UniversalClassifierModel(num_classes=9).eval()
NET.load_state_dict(load("./models/UniversalClassifier/multiClassifierCheckpoint.pth.tar", map_location ='cpu')["state_dict"])


def PredictDisease(img_bytes):
    img_np = np.array(Image.open(BytesIO(img_bytes)).convert("RGB")) # Convert image bytes into a matrix
    img_tensor = TRANSFORMS(img_np)

    prediction = NET(img_tensor.unsqueeze(0))
    prediction = softmax(prediction, dim=1).squeeze().detach().numpy()

    label_idx = np.argmax(prediction)

    return CLASSES[label_idx]