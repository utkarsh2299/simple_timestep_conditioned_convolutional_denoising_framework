import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from scipy import signal
import soundfile as sf

class STFT(nn.Module):
    def __init__(self, n_fft=4096, hop_length=1024, win_length=4096):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)
    
    def forward(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.win_length, window=self.window,
                         return_complex=True)

class STFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFT()
        
    def forward(self, x, y):
        x_stft = self.stft(x)
        y_stft = self.stft(y)
        
        mag_loss = torch.mean(torch.abs(torch.abs(x_stft) - torch.abs(y_stft)))
        phase_loss = torch.mean(torch.abs(1 - torch.cos(torch.angle(x_stft) - torch.angle(y_stft))))
        
        return mag_loss + 0.5 * phase_loss

def generate_noisy_audio(clean_audio_path, output_dir, noise_factor=0.1, snr_db=15):
    """Generate noisy version of clean audio with controlled SNR"""
    audio, sr = librosa.load(clean_audio_path, sr=48000)
    
    # Add noise
    noise = np.random.normal(0, 1, len(audio))
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor for desired SNR
    scaling_factor = np.sqrt(signal_power / (noise_power * (10 ** (snr_db/10))))
    noise_scaled = noise * scaling_factor * noise_factor
    
    noisy_audio = audio + noise_scaled
    
    # Normalize
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    
    os.makedirs(output_dir, exist_ok=True)
    noisy_audio_path = os.path.join(output_dir, f"noisy_{os.path.basename(clean_audio_path)}")
    sf.write(noisy_audio_path, noisy_audio, sr)
    
    return noisy_audio_path

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, segment_length=48000):
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')]
        self.noisy_files = [os.path.join(noisy_dir, f"noisy_{os.path.basename(f)}") for f in self.clean_files]
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_audio, _ = librosa.load(self.clean_files[idx], sr=48000)
        noisy_audio, _ = librosa.load(self.noisy_files[idx], sr=48000)
        
        # Random segment for training
        if len(clean_audio) > self.segment_length:
            start = np.random.randint(0, len(clean_audio) - self.segment_length)
            clean_audio = clean_audio[start:start + self.segment_length]
            noisy_audio = noisy_audio[start:start + self.segment_length]
        else:
            # Pad if audio is shorter than segment length
            clean_audio = np.pad(clean_audio, (0, self.segment_length - len(clean_audio)))
            noisy_audio = np.pad(noisy_audio, (0, self.segment_length - len(noisy_audio)))
        
        return (torch.FloatTensor(noisy_audio), torch.FloatTensor(clean_audio))

class AudioDiffusionModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        
        # Add dropout to encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 256, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 512, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
        )
        
        # Add dropout to decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(256, 128, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(128, 64, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 1, 15, stride=1, padding=7),
        )
        
    def forward(self, x, t_emb):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        
        # Time embedding
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        
        # Combine features
        x = x + t_emb
        
        x = self.decoder(x)
        return x.squeeze(1)  # Remove channel dimension

def sinusoidal_embedding(timesteps, dim, device):
    """
    Generate sinusoidal embeddings with proper device handling.
    """
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def train_diffusion_model(
    clean_dir,
    noisy_dir,
    epochs=1000,
    batch_size=32,
    base_lr=1e-4,
    weight_decay=1e-2,
    checkpoint_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = AudioDataset(clean_dir, noisy_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model with dropout
    model = AudioDiffusionModel(dropout_rate=0.1).to(device)
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Calculate total steps for OneCycleLR
    total_steps = epochs * len(dataloader)
    
    # Warmup scheduler: Linear warmup for first 5% of training
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0,
        total_iters=int(0.05 * total_steps)
    )
    
    # Main scheduler: OneCycleLR for cyclical learning rate
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of time in ascending phase
        div_factor=10,  # Initial learning rate is max_lr/10
        final_div_factor=1000,  # Final learning rate is max_lr/1000
    )
    
    stft_loss = STFTLoss().to(device)
    l1_loss = nn.L1Loss()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    warmup_period = int(0.05 * total_steps)  # 5% of total steps
    step_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            
            timesteps = torch.randint(0, 1000, (noisy.shape[0],), device=device)
            t_emb = sinusoidal_embedding(timesteps, 128, device)
            
            # Forward pass with dropout enabled during training
            predicted = model(noisy, t_emb)
            
            # Combined loss with L1 regularization
            reconstruction_loss = stft_loss(predicted, clean) + l1_loss(predicted, clean)
            
            # Add L2 regularization loss (already handled by weight_decay in optimizer)
            loss = reconstruction_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            if step_count < warmup_period:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            step_count += 1
            epoch_loss += loss.item()
            
            # Print current learning rate periodically
            if step_count % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step_count}, Current LR: {current_lr:.2e}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{datetime.now():%Y%m%d_%H%M%S}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

def denoise_audio(model_path, noisy_audio_path, output_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Denoise audio using trained model"""
    model = AudioDiffusionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and process audio
    audio, sr = librosa.load(noisy_audio_path, sr=48000)
    audio_tensor = torch.FloatTensor(audio).to(device)
    
    # Process in segments to avoid memory issues
    segment_length = 48000
    denoised_segments = []
    
    with torch.no_grad():
        for i in range(0, len(audio), segment_length):
            segment = audio_tensor[i:i+segment_length]
            if len(segment) < segment_length:
                segment = torch.nn.functional.pad(segment, (0, segment_length - len(segment)))
            
            # Zero timestep for inference
            t_emb = sinusoidal_embedding(torch.zeros(1, device=device, dtype=torch.long), 128).to(device)
            
            denoised_segment = model(segment.unsqueeze(0), t_emb).squeeze(0)
            denoised_segments.append(denoised_segment.cpu().numpy())
    
    # Combine segments
    denoised_audio = np.concatenate(denoised_segments)[:len(audio)]
    
    # Save denoised audio
    sf.write(output_path, denoised_audio, sr)

if __name__ == "__main__":
    clean_dir ="gt_audio"
    noisy_dir = "gl_audio"
    
    # Generate noisy audio if needed
    if not os.path.exists(noisy_dir):
        print("inside noiseuy")
        os.makedirs(noisy_dir, exist_ok=True)
        for file_name in os.listdir(clean_dir):
            if file_name.endswith(".wav"):
                clean_path = os.path.join(clean_dir, file_name)
                generate_noisy_audio(clean_path, noisy_dir)
    
    # Train model
    train_diffusion_model(clean_dir, noisy_dir)

    # clean_audio_dir = "/speech/utkarsh/database/hindi_male_mono/wav"
    # noisy_audio_dir = "./noisy_audio"