# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .mtr_decoder import MTRDecoder


__all__ = {
    'MTRDecoder': MTRDecoder
}


def build_motion_decoder(in_channels, config, training_gan=False, tb_writer=None):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config,
        training_gan=training_gan,
        tb_writer=tb_writer
    )

    return model
