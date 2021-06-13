import torch.nn as nn
import timm
import config
import pretrainedmodels


class Model(nn.Module):
    def __init__(self, training = True, pretrained = True ):
        super().__init__()
        self.training = training

        self.model = timm.create_model(config.MODEL_NAME,
                                            pretrained=pretrained, in_chans=config.CHANNELS)
        if config.MODEL_NAME in ['resnet18', 'resnet18d']:
            self.in_features = self.model.fc.in_features
        else:
            self.in_features = self.model.classifier.out_features
        self.model.fc = nn.Linear(self.in_features, config.TARGET_SIZE)
#         print(self.model)
#         print(f'\nUsing {config.MODEL_NAME}, model output layer: {self.output_layer}\n')

    def forward(self, x):
        return self.model(x)

# class Model(nn.Module):
#     def __init__(self, training = True, pretrained = True ):
#         super().__init__()
#         self.training = training
#         self.model = timm.create_model(config.MODEL_NAME,
#                                             pretrained=pretrained, in_chans=config.CHANNELS)
        
#         self.n_features = self.model.classifier.out_features
#         self.output_layer = nn.Linear(self.n_features, config.TARGET_SIZE)
#         print(f'\nUsing {config.MODEL_NAME}, model output layer: {self.model.classifier.out_features}\n')

#         self.p_dropout = 0.5

#     def forward(self, x):
#         model_last_layer = self.model(x)
#         output = self.output_layer(model_last_layer)
# #         for i in range(5):
# #             if i == 0:
# #                 output_msd = self.output_layer(nn.functional.dropout(model_last_layer, p=self.p_dropout, training=self.training))
# #             else:
# #                 output_msd += self.output_layer(nn.functional.dropout(model_last_layer, p=self.p_dropout, training=self.training))
# #         output_msd = output_msd / 5
#         return output
