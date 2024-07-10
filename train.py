from simple_unet import Simple_UNet
from noise_scheduler import forward_diffusion_sample
import torch.nn.functional as F
import torch

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'

model = Simple_UNet()

def loss(model, x_0, t):
    noisy_sample, noise = forward_diffusion_sample(x_0, t, DEVICE)
    y = model(noisy_sample, t) # need to revise model to understand timestamps
    return F.l1_loss(y, noise)
