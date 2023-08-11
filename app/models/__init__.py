from .LiverClassifier.Model import LiverClassifier
from .UniversalClassifier.Model import UniversalClassifier
from .KneeXRayClassifier.Model import KneeXRayClassifier

import torch.nn.functional as F

import numpy as np

universalClassifier = UniversalClassifier().eval()
universalClassifier.load_weights()

liverModel = LiverClassifier().eval()
liverModel.load_weights()

kneeModel = KneeXRayClassifier()


def PredictImageType(img):
    y_pred = universalClassifier.forward(img)

    y_pred = F.softmax(y_pred, dim=1).squeeze().detach().numpy()
    label_idx = np.argmax(y_pred)

    return universalClassifier.class_names[label_idx], label_idx


def PredictDisease(img, index):
    if index == 5:
        pred_tensor = kneeModel.forward(img)
        pred_array = np.around(pred_tensor[0][0] * 100, 2)

        dic = {
            0: [0, pred_array],
            1: [1, 100 - pred_array]
        }

        return dic
    
    elif index == 6:
        return model_logic(model=liverModel, img=img)
    else:
        return model_logic(model=liverModel, img=img)
    

def model_logic(model, img):
    pred_tensor = model.forward(img)
    pred = F.softmax(pred_tensor, dim=1).squeeze().detach().numpy()

    results = [[i, p] for i, p in enumerate(pred*100)]
    results = np.around(results, 2)
    results = results[results[:, 1].argsort()][::-1]

    dic = {i:[int(x), y] for i, (x, y) in enumerate(results.tolist())}

    return dic