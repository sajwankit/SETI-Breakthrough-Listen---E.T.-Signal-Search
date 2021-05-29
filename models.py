import torch.nn as nn
import timm
import config

class Model(nn.Module):
    def __init__(self, training = True, pretrained = True, target_size = 1 ):
        super().__init__()
        self.training = training
        self.target_size = target_size
        self.model = timm.create_model(config.MODEL_NAME,
                                            pretrained=pretrained, in_chans=config.CHANNELS)
        self.in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(self.in_features, 1024)
        self.fc = nn.Linear(1024, config.TARGET_SIZE)
        self.p_dropout = 0.5

    def forward(self, x):
        output = self.model(x)
        for i in range(5):
            if i == 0:
                h = self.fc(nn.functional.dropout(output, p=self.p_dropout, training=self.training))
            else:
                h += self.fc(nn.functional.dropout(output, p=self.p_dropout, training=self.training))
        h = h / len(self.dropouts)
        return h

    # def __init__(self, pretrained = True, target_size = 1 ):
    #     super().__init__()
    #     self.target_size = target_size
    #     self.model = timm.create_model(config.MODEL_NAME,
    #                                         pretrained=pretrained, in_chans=config.CHANNELS)
        
    #     self.dropouts = nn.ModuleList([nn.Dropout(p = 0.5) for _ in range(5)])
    #     self.n_features = self.model.head.fc.in_features
    #     # self.model.classifier = nn.Linear(self.n_features, 1024)  
    #     self.fc = nn.Linear(self.n_features, config.TARGET_SIZE)
    #     self.model.head.fc = nn.Linear(self.n_features, 1024)

    # def forward(self, x):
    #     last_layer = self.model(x).fc
    #     for i, dropout in enumerate(self.dropouts):
    #         if i == 0:
    #             h = self.fc(dropout(last_layer.fc))
    #         else:
    #             h += self.fc(dropout(last_layer.fc))
    #     h = h / len(self.dropouts)
    #     return h



# class NeuralNet(nn.Module):
#     def __init__(self, hidden_size=768, num_class=2):
#         super(NeuralNet, self).__init__()

#         self.bert = BertModel.from_pretrained('bert-base-uncased',  
#                                         output_hidden_states=True,
#                                         output_attentions=True)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.weights = nn.Parameter(torch.rand(13, 1))
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(0.5) for _ in range(5)
#         ])
#         self.fc = nn.Linear(hidden_size, num_class)

#     def forward(self, input_ids, input_mask, segment_ids):
#         all_hidden_states, all_attentions = self.bert(input_ids, token_type_ids=segment_ids,
#                                                                 attention_mask=input_mask)[-2:]
#         batch_size = input_ids.shape[0]
#         ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
#             13, batch_size, 1, 768)
#         atten = torch.sum(ht_cls * self.weights.view(
#             13, 1, 1, 1), dim=[1, 3])
#         atten = F.softmax(atten.view(-1), dim=0)
#         feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#         for i, dropout in enumerate(self.dropouts):
#             if i == 0:
#                 h = self.fc(dropout(feature))
#             else:
#                 h += self.fc(dropout(feature))
#         h = h / len(self.dropouts)
#         return h

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