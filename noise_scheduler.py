import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# get data
yesno_data = torchaudio.datasets.YESNO('.', download=True) # more about YESNO -> https://www.openslr.org/1
data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1)
signal_1 = data_loader.dataset[0][0]
signal_1_samplerate = data_loader.dataset[0][1]
# data_loader.dataset -- 0 is the tensor of audio, 1 is the samplerate, 2 is the sequence of "yes"s and "no"s.

# indexer helper function

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    batching_dimensions = (1,) * (len(x_shape) - 1)
    return out.reshape(batch_size, *batching_dimensions).to(t.device)



# convert tensor to audio
def tensor_to_audio(tensor, samplerate, filepath="output.wav"):
    torchaudio.save(filepath, tensor, samplerate)
    # if using a colab notebook:
    # from IPython.display import Audio, display
    # display(Audio(filepath, autoplay=True))


def forward_diffusion_sample(x_0, t, device):
    # retrieve values
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    breakpoint;
    # send to gpu or mps to increase processing speed
    noise.to(device)
    sqrt_alphas_cumprod_t.to(device)
    sqrt_one_minus_alphas_cumprod_t.to(device)

    noisy_sample = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    """ What's happening here?

        sqrt_alphas_cumprod_t * x_0 -> 
            Recall that `x_0` is the original signal. 
            `sqrt_alphas_cumprod_t` represets the x of signal that has not been degraded.
            We element-wise multiply them to create signal that has been degraded to the degree of coefficient sqrt_alphas_cumprod_t.

        sqrt_one_minus_alphas_cumprod_t * noise ->
            `sqrt_one_minus_alphas_cumprod_t` represents the amount of signal that is going to be degraded.
            We element-wise multiply it with noise to get a tensor of noise

        summing the two ->
            element-wise summation results in a signal made up of sqrt_alphas_cumprod_t signal and sqrt_one_minus_alphas_cumprod_t noise
    """

    return (noisy_sample, noise)



def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# beta schedule
timesteps = 1000
betas = linear_beta_schedule(timesteps)

# identity definitions
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'


t = torch.Tensor([1]).type(torch.int64)

signal, noise = forward_diffusion_sample(signal_1, t, DEVICE)
tensor_to_audio(signal, signal_1_samplerate)



# see noise in mel spectrogram
# transform = torchaudio.transforms.MelSpectrogram(signal_1_samplerate)
# mel = transform(signal)[0]

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
plt.suptitle('Noise added to Mel Spectrograms', fontsize=16)
plt.ylabel('Frequency bins')
plt.xlabel('Time frames')

num_images = 10
stepsize = timesteps // num_images
transform = torchaudio.transforms.MelSpectrogram(signal_1_samplerate)

for i in range(0, timesteps, stepsize):
    t = torch.Tensor([i]).type(torch.int64)
    signal, noise = forward_diffusion_sample(signal_1, t, DEVICE)
    tensor_to_audio(signal, signal_1_samplerate, f'out{i}.wav')
    # see noise in mel spectrogram
    mel = transform(signal)[0]
    plt.subplot(1, num_images+1, i//stepsize + 1)
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.title(f'Timestep {i}')
    plt.axis('off')


cax = plt.axes([0.92, 0.1, 0.02, 0.8])
plt.colorbar(cax=cax, format='%+2.0f dB')

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()