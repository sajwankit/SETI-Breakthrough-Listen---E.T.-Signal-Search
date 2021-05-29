import torch.nn as nn
import pretrainedmodels
import timm
import config

class Model(nn.Module):

    def __init__(self, pretrained = True, target_size = 1 ):
        super().__init__()
        self.target_size = target_size
        self.basemodel = timm.create_model(config.MODEL_NAME,
                                            pretrained=pretrained, in_chans=config.CHANNELS)
        self.in_features = self.basemodel.head.fc.in_features
        self.model = self.basemodel
        self.dropouts = nn.ModuleList([ nn.Droupout(0.5) for _ in range(5)])
        self.model.head.fc = nn.Linear(self.in_features, config.TARGET_SIZE)
        

    def forward(self, x):
        output = self.model(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.linear(dropout(output))
            else:
                h += self.linear(dropout(output))
             
        return output

# def get_model(pretrained):
#     if pretrained:
#         model = pretrainedmodels.__dict__['alexnet'](pretrained = 'imagenet')
#     else:
#         model = pretrainedmodels.__dict__['alexnet'](pretrained = None)

#     model.last_linear = nn.Sequential(
#                                     nn.BatchNorm1d(4096),
#                                     nn.Dropout(p = 0.25),
#                                     nn.Linear(in_features = 4096, out_features = 2048),
#                                     nn.ReLU(),
#                                     nn.BatchNorm1d(2048, eps = 1e-05, momentum = 0.1),
#                                     nn.Dropout(p = 0.5),
#                                     nn.Linear(in_features = 2048, out_features = 1)
#                                     )
#     return model
