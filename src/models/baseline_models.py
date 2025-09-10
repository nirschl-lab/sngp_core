import torch
import torch.nn as nn
from typing import Iterable, Optional
from src.utils.helper import count_num_parms

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)
    

# (256, 256, 128, 1024) -> 37544
# (256, 256, 1024) -> 435720
# (256, 256, 128) -> 198280
# (256, 128, 128, 32) -> 147905

class BaselineModel(nn.Module):
    def __init__(self, in_dim=384, num_classes=8, hidden: Iterable[int] = (256, 128, 128, 32), p_drop: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, masks=None):
        return self.net(x)
    
    
if __name__ == '__main__':
    base_model = BaselineModel(in_dim=384, num_classes=8)
    print('Base Model - ' + count_num_parms(base_model))