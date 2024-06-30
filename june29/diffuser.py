import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)

        self.time_embed = nn.Linear(1,256)

    def forward(self, x, t):
        # Embed the timestep
        t_embed = self.time_embed(t.float().unsqueeze(-1)).unsqueeze(-1)
        
        x1 = F.relu(self.enc1(x) + t_embed[:, :64, :])
        x2 = F.relu(self.enc2(x1) + t_embed[:, :128, :])
        x3 = F.relu(self.enc3(x2) + t_embed[:, :256, :])
        x = F.relu(self.dec1(x3) + t_embed[:, :128, :])
        x = F.relu(self.dec2(x) + t_embed[:, :64, :])
        x = torch.tanh(self.dec3(x) + t_embed[:, :1, :])
        return x

class DiffusionModel:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t]).unsqueeze(1).unsqueeze(2).expand_as(x0)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hats[t]).unsqueeze(1).unsqueeze(2).expand_as(x0)
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
        return xt, noise

    def reverse_diffusion(self, xt, t):
        return self.model(xt, t)


from torch.utils.data import DataLoader, Dataset

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_model(model, diffusion_model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.unsqueeze(1).float()
            t = torch.randint(0, diffusion_model.timesteps, (batch.size(0),), device=batch.device)
            xt, noise = diffusion_model.forward_diffusion(batch, t)
            pred_noise = diffusion_model.reverse_diffusion(xt, t)
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def perform_inference(model, diffusion_model, initial_noise, timesteps=1000):
    model.eval()
    with torch.no_grad():
        xt = initial_noise
        for t in reversed(range(timesteps)):
            timestep = torch.full((xt.size(0),), t, device=xt.device, dtype=torch.long)
            xt = diffusion_model.reverse_diffusion(xt, timestep)
            alpha_hat_t = diffusion_model.alpha_hats[t]
            # Ensure the alpha_hat_t tensor is the correct shape for broadcasting
            sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t).reshape(1, 1, 1)
            sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat_t).reshape(1, 1, 1)
            xt = (xt - sqrt_one_minus_alpha_hat_t * torch.randn_like(xt)) / sqrt_alpha_hat_t
    return xt



if __name__ == "__main__":

    device = 'cpu'
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    # Generate some random audio data for demonstration
    audio_data = [torch.randn(16000) for _ in range(100)]
    dataset = AudioDataset(audio_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet()
    diffusion_model = DiffusionModel(model)

    train_model(model, diffusion_model, dataloader, epochs=2)

    initial_noise = torch.randn(1, 1, 16000)
    generated_signal = perform_inference(model, diffusion_model, initial_noise)
    # torchaudio.io.play_audio(generated_signal)