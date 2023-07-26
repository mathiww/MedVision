from .LiverClassifier.Model import LiverClassifier
from .UniversalClassifier.Model import UniversalClassifier
import torch.nn.functional as F

import numpy as np

universalClassifier = UniversalClassifier().eval()
universalClassifier.load_weights()

liverModel = LiverClassifier().eval()
liverModel.load_weights()


def PredictImageType(img):
    y_pred = universalClassifier.forward(img)

    y_pred = F.softmax(y_pred, dim=1).squeeze().detach().numpy()
    label_idx = np.argmax(y_pred)

    return universalClassifier.class_names[label_idx], label_idx


def PredictDisease(img, index):
    if index == 6:
        return model_logic(model=liverModel, img=img)
    else:
        return model_logic(model=liverModel, img=img)


def model_logic(model, img):
    pred_tensor = model.forward(img)
    pred = F.softmax(pred_tensor, dim=1).squeeze().detach().numpy()

    results = [[i, np.round(p, decimals=4)*100] for i, p in enumerate(pred)]
    results = np.array(results)
    results = results[results[:, 1].argsort()][::-1]

    dic = {i:r for i, r in enumerate(results.tolist())}

    return dic