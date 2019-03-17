# A Style Based Variational Autoencoder
[![Build Status](https://travis-ci.org/orgoro/style-vae.svg?branch=master)](https://travis-ci.org/orgoro/style-vae)

## Architecture
![arch](style_vae/doc/arch.png)

## Loss
The loss is comprised out of two components:
* **Reconstruction Loss** - based on perceptual loss (pre-trained VGG16 features)
* **Latent Loss** - kl-divergence loss 

## Reconstruction Results
* **512 params:**
![512](style_vae/doc/512.png)
* **100 params:**
![100](style_vae/doc/100.png)
