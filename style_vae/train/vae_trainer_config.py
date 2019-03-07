from dataclasses import dataclass


@dataclass
class VaeTrainerConfig:
    name: str = 'default-vae'
    reload_model: bool = False
    save_model: bool = True
    batch_size: int = 64
    num_epochs: int = 500
    lr: float = 1e-5
    recon_loss: str = 'perceptual'
    latent_weight: float = 0.5

    def __str__(self):
        res = 'VaeTrainerConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


if __name__ == '__main__':
    x = VaeTrainerConfig()
    print(x)
