from dataclasses import dataclass
from os import path


@dataclass
class VaeTrainerConfig:
    name: str = 'default-vae'
    reload_model: bool = True
    save_model: bool = True
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 2e-5
    recon_loss: str = 'perceptual'
    latent_weight: float = 2.
    data_regex: str = path.join('/data', 'ffhq_256', '*.png')

    def __str__(self):
        res = 'VaeTrainerConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


if __name__ == '__main__':
    x = VaeTrainerConfig()
    print(x)
