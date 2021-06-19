import torch.nn as nn
import timm
import config
import pretrainedmodels


class Model(nn.Module):
    def __init__(self, pretrained = True ):
        super().__init__()
        self.base_model = timm.create_model(config.MODEL_NAME,
                                            pretrained=pretrained, in_chans=config.CHANNELS)
        if config.MODEL_NAME in ['resnet18', 'resnet18d']:
            self.in_features = self.base_model.fc.in_features
        else:
            self.in_features = self.base_model.classifier.out_features
            
            
        if config.DROPOUT:
            self.base_model_out_features = 1024*10
            self.base_model.fc = nn.Linear(self.in_features, self.base_model_out_features)
            self.output_layer = nn.Linear(self.base_model_out_features, config.TARGET_SIZE)
        else:
            self.base_model_out_features = config.TARGET_SIZE
            self.base_model.fc = nn.Linear(self.in_features, self.base_model_out_features)
            self.output_layer = self.base_model.fc
        
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.7) for _ in range(16)
        ])
#         print(self.model)
        print(f'\nUsing {config.MODEL_NAME}, model output layer: {self.output_layer} with DROPOUT {config.DROPOUT}\n')

    def forward(self, x):
        base_model_logits = self.base_model(x)
        if config.DROPOUT:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    logits = self.output_layer(dropout(base_model_out))
                else:
                    logits += self.output_layer(dropout(base_model_out))
            logits /= len(self.dropouts)
            return logits
        else:
            return base_model_logits
        
        
#         output = self.output_layer(model_last_layer)

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
