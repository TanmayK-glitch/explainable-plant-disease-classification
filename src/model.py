import torch 
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

def get_model(num_classes=15):

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # print(model)

    input_feat = model.classifier[1].in_features

    # for params in model.parameters():
    #     params.requires_grad = False

    ## Unfreeze last 3 layers (fine-tune)
    for idx, block in enumerate(model.features):
        if idx >= 5:
            for param in model.parameters():
                param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(input_feat, num_classes)
    )

    return model