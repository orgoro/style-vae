from dataclasses import dataclass


@dataclass
class VaeTrainerConfig:
    name: str = 'default-vae'
    reload_model: bool = True
    save_model: bool = True
    batch_size: int = 32
    num_epochs: int = 20
    lr: float = 5e-5
    recon_loss: str = 'perceptual'
    latent_weight: float = 10

    def __str__(self):
        res = 'VaeTrainerConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


if __name__ == '__main__':
    x = VaeTrainerConfig()
    print(x)
