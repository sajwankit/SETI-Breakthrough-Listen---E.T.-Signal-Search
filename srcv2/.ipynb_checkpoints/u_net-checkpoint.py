from segmentation_models_pytorch import Unet
import torch.nn as nn
import torch
import config
from typing import Optional, Union, List

class myUnet(Unet):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,

        ):
        super().__init__(
        encoder_name,
        encoder_depth,
        encoder_weights,
        decoder_use_batchnorm,
        decoder_channels,
        decoder_attention_type,
        in_channels,
        classes,
        activation,
        aux_params)

        self.decoder_final_layer = nn.Sequential(
                                    nn.ConvTranspose2d(
                                                        in_channels=16,
                                                        out_channels=config.CHANNELS,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1
                                                        ),
                                    nn.BatchNorm2d(config.CHANNELS),
                                    )


    def forward(self, x):
        xon = x[:, 0,: ,: ].reshape(-1, 1, x[:, 0,: ,: ].shape[1], x[:, 0,: ,: ].shape[2])
        xoff = x[:, 1,: ,: ].reshape(-1, 1, x[:, 1,: ,: ].shape[1], x[:, 1,: ,: ].shape[2])
#         print(x0.shape, x1.shape)
        encoder_input = nn.functional.interpolate(xoff, size=(256, 256))
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(encoder_input)
        decoder_output = self.decoder(*features)
        recon_xon = self.decoder_final_layer(decoder_output)
        recon_xon = nn.functional.interpolate(recon_xon, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        recon_xon = nn.functional.instance_norm(recon_xon)
        return recon_xon, xon, torch.randn(1), torch.randn(1), torch.randn(1)
