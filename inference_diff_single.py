import torch
import librosa
import soundfile as sf
import numpy as np
# import os
from tqdm import tqdm
import numpy as np
import librosa
import torch
import torch.nn as nn
import scipy
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from datetime import datetime
from scipy import signal
import soundfile as sf
device="cuda" if torch.cuda.is_available() else "cpu"
class AudioDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 15, stride=2, padding=7, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, 15, stride=1, padding=7),
        )
        
    def forward(self, x, t_emb):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x + t_emb
        x = self.decoder(x)
        return x.squeeze(1)

def sinusoidal_embedding(timesteps, dim, device=device):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def denoise_audio_file(
    model_path,
    input_audio_path,
    output_audio_path,
    segment_length=48000*4,  # 4 seconds at 48kHz
    overlap_ratio=0.25,      # 25% overlap
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Denoise audio with improved overlap handling and crossfading
    """
    # Load model
    model = AudioDiffusionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load audio
    audio, sr = librosa.load(input_audio_path, sr=48000)
    print(f"Processing audio of length: {len(audio)} samples ({len(audio)/sr:.2f} seconds)")
    
    # Calculate overlap samples
    overlap_samples = int(segment_length * overlap_ratio)
    hop_length = segment_length - overlap_samples
    
    # Create crossfade windows
    fade_in = np.linspace(0, 1, overlap_samples)
    fade_out = np.linspace(1, 0, overlap_samples)
    
    # Create hanning window for the full segment
    full_window = np.hanning(segment_length)
    
    # Calculate number of segments
    num_segments = int(np.ceil((len(audio) - overlap_samples) / hop_length))
    
    # Initialize output array
    denoised_audio = np.zeros(len(audio) + segment_length)  # Add padding
    weights = np.zeros_like(denoised_audio)  # For weighted averaging
    
    print(f"Processing {num_segments} segments with {overlap_samples} samples overlap")
    
    with torch.no_grad():
        for i in tqdm(range(num_segments)):
            # Calculate segment indices
            start_idx = i * hop_length
            end_idx = start_idx + segment_length
            
            # Extract and pad segment if necessary
            segment = audio[start_idx:min(end_idx, len(audio))]
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            # Convert to tensor
            segment_tensor = torch.FloatTensor(segment).to(device)
            
            # Get time embedding (use 0 for inference)
            t_emb = sinusoidal_embedding(torch.zeros(1, device=device, dtype=torch.long), 128, device)
            
            # Denoise segment
            denoised_segment = model(segment_tensor.unsqueeze(0), t_emb).squeeze(0).cpu().numpy()
            
            # Apply window to reduce edge effects
            denoised_segment = denoised_segment * full_window
            
            # Add to output with weighted overlap
            denoised_audio[start_idx:start_idx + segment_length] += denoised_segment
            weights[start_idx:start_idx + segment_length] += full_window
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    denoised_audio = denoised_audio / weights
    
    # Trim any extra padding
    denoised_audio = denoised_audio[:len(audio)]
    
    # Apply final normalization
    denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
    
    # Optional: Apply subtle noise reduction as post-processing
    if True:  # You can make this a parameter
        denoised_audio = post_process(denoised_audio, sr)
    
    # Save denoised audio
    sf.write(output_audio_path, denoised_audio, sr)
    print(f"Saved denoised audio to: {output_audio_path}")
    
    return denoised_audio

def post_process(audio, sr):
    """
    Apply subtle post-processing to reduce any remaining artifacts
    """
    # Spectral gating
    D = librosa.stft(audio, n_fft=4096, hop_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D))
    
    # Calculate threshold for each frequency bin
    threshold = np.mean(S_db, axis=1, keepdims=True) - 2 * np.std(S_db, axis=1, keepdims=True)
    mask = S_db > threshold
    
    # Apply soft mask
    mask = mask.astype(np.float32)
    smoothed_mask = scipy.ndimage.gaussian_filter(mask, sigma=[2, 1])
    
    # Apply mask and reconstruct
    D_cleaned = D * smoothed_mask
    audio_cleaned = librosa.istft(D_cleaned, hop_length=1024)
    
    # Trim to original length
    audio_cleaned = audio_cleaned[:len(audio)]
    
    return audio_cleaned

# Usage example
if __name__ == "__main__":
    model_path = "checkpoints/final2.pth"
    input_audio = "test.wav"
    output_audio = "denoised_audio_testttt.wav"
    
    denoise_audio_file(
        model_path=model_path,
        input_audio_path=input_audio,
        output_audio_path=output_audio,
        segment_length=48000*4,  # 4 seconds segments
        overlap_ratio=0.25       # 25% overlap
    )
