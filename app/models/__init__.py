from .UniversalClassifier.Model import UniversalClassifier
from .BrainTumorClassifier.Model import BrainTumorClassifier
from .KneeXRayClassifier.Model import KneeXRayClassifier
from .LiverClassifier.Model import LiverClassifier
from .EyeClassifier.Model import EyeClassifier

import torch.nn.functional as F
import numpy as np


universalClassifier = UniversalClassifier().eval()
universalClassifier.load_weights()

brainMRIModel = BrainTumorClassifier().eval()
brainMRIModel.load_weights()

kneeXRModel = KneeXRayClassifier()

liverMRIModel = LiverClassifier().eval()
liverMRIModel.load_weights()

eyeModel = EyeClassifier().eval()
eyeModel.load_weights()


"""
Indexes
0 - Blood
1 - Brain MRI
2 - Chest CT
3 - Chest XR
4 - Knee MRI
5 - Knee XR
6 - Liver MRI
7 - Eye
8 - Non medical image
"""


def PredictImageType(img):
    y_pred = universalClassifier.forward(img)

    y_pred = F.softmax(y_pred, dim=1).squeeze().detach().numpy()
    label_idx = np.argmax(y_pred)

    return universalClassifier.class_names[label_idx], label_idx


def PredictDisease(img, index):
    # if index == 0:
    #     return ModelLogic(model=bloodModel, img=img)
    if index == 1:
        return ModelLogic(model=brainMRIModel, img=img)
    # elif index == 2:
    #     return ModelLogic(model=chestCTModel, img=img)
    # elif index == 3:
    #     return ModelLogic(model=chestXRModel, img=img)
    # elif index == 4:
    #     return ModelLogic(model=kneeMRIModel, img=img)
    elif index == 5:
        return BinaryModelLogic(model=kneeXRModel, img=img)
    elif index == 6:
        return ModelLogic(model=liverMRIModel, img=img)
    elif index == 7:
        return ModelLogic(model=eyeModel, img=img)
    else:
        return ModelLogic(model=liverMRIModel, img=img)
    

def ModelLogic(model, img):
    pred_tensor = model.forward(img)
    pred = F.softmax(pred_tensor, dim=1).squeeze().detach().numpy()

    results = [[i, p] for i, p in enumerate(pred*100)]
    results = np.around(results, 1)
    results = results[results[:, 1].argsort()][::-1]

    dic = {i:[model.class_names[int(x)], y] for i, (x, y) in enumerate(results.tolist())}

    return dic


def BinaryModelLogic(model, img):
    pred = model.forward(img)

    pred = np.around(pred[0][0] * 100, 1)
    pred_array = np.array([[1, pred], [0, 100 - pred]])
    sorted_pred_array = pred_array[pred_array[:, 1].argsort()][::-1]
    

    dic = {i:[model.class_names[int(x)], y] for i, (x, y) in enumerate(sorted_pred_array.tolist())}

    return dic