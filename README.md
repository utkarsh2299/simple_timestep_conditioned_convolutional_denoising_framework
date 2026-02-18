# 🎧 Audio Diffusion-Based Denoising Model (PyTorch)

A PyTorch-based **audio denoising system** inspired by diffusion modeling principles.
This project trains a convolutional encoder–decoder network with sinusoidal time embeddings to reconstruct clean audio from noisy inputs using combined STFT and L1 losses.

---

##  Features

*  Controlled noisy audio generation with configurable SNR
*  STFT-based magnitude + phase loss
*  Diffusion-style sinusoidal timestep embeddings
*  Conv1D Encoder–Decoder architecture with dropout
*  Warmup + OneCycleLR learning rate scheduling
*  Gradient clipping for stable training
*  Automatic best-checkpoint saving
*  Segment-wise inference for long audio

---

##  Model Architecture

The model consists of:

### Encoder

* 1D Convolution layers
* Strided downsampling
* LeakyReLU activations
* Dropout regularization

### Time Embedding

* Sinusoidal embedding (diffusion-style)
* MLP projection to latent dimension

### Decoder

* ConvTranspose1D upsampling layers
* Skip-less reconstruction
* Final waveform prediction

---

##  Project Structure

```
.
├── train_diff_new.py
├── gt_audio/              # Clean audio files (.wav)
├── gl_audio/              # Generated noisy audio
├── checkpoints/           # Saved best models
└── README.md
```

---

##  Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/audio-diffusion-denoiser.git
cd audio-diffusion-denoiser
```

### 2️⃣ Create environment

```bash
conda create -n audio_diff python=3.10
conda activate audio_diff
```

### 3️⃣ Install dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scipy soundfile
```

---

##  Dataset Preparation

Place your clean `.wav` files inside:

```
gt_audio/
```

* Sampling rate: **48 kHz**
* Mono audio recommended

If `gl_audio/` does not exist, the script will automatically:

* Generate Gaussian noise
* Apply controlled SNR
* Save noisy versions as:

  ```
  noisy_filename.wav
  ```

---

##  Training

Run:

```bash
python train_diff_new.py
```

### Default Training Parameters

| Parameter      | Value                 |
| -------------- | --------------------- |
| Epochs         | 1000                  |
| Batch Size     | 32                    |
| Base LR        | 1e-4                  |
| Weight Decay   | 1e-2                  |
| Segment Length | 48000 (1 sec @ 48kHz) |

---

##  Loss Function

Training uses a combination of:

### 1️⃣ STFT Loss

* Magnitude difference
* Phase consistency loss

### 2️⃣ L1 Waveform Loss

Final loss:

```
Loss = STFT_Loss + L1_Loss
```

Weight decay in AdamW handles L2 regularization.

---

##  Learning Rate Strategy

Training uses a two-stage scheduler:

1. **Linear Warmup (5%)**
2. **OneCycleLR**

   * 30% ascending phase
   * LR decay to 1/1000th final value

This stabilizes early training and improves convergence.

---

##  Checkpoints

Best model (lowest validation loss) is automatically saved:

```
checkpoints/best_model_YYYYMMDD_HHMMSS.pth
```

Saved state includes:

* Model weights
* Optimizer state
* Scheduler state
* Epoch
* Loss

---

##  Inference (Denoising)

Use:

```python
from train_diff_new import denoise_audio

denoise_audio(
    model_path="checkpoints/best_model_xxx.pth",
    noisy_audio_path="noisy.wav",
    output_path="denoised.wav"
)
```

### Inference Details:

* Processes audio in 1-second segments
* Uses timestep = 0 embedding
* Concatenates reconstructed segments

---

##  Diffusion Design Notes

Although simplified compared to full DDPMs:

* Uses timestep conditioning
* Random timestep sampling during training
* Sinusoidal embeddings
* Learns direct waveform reconstruction

This behaves like a **conditional denoising diffusion-inspired network** rather than a full iterative diffusion sampler.

---

##  Recommended Improvements

Future work ideas:

* Multi-resolution STFT loss
* U-Net skip connections
* True diffusion noise schedule
* EMA weight averaging
* Mixed precision training
* Spectral normalization
* Perceptual audio metrics (PESQ, STOI)

---

##  Hardware Recommendations

* GPU recommended (CUDA)
* 8GB+ VRAM suggested
* CPU training supported but slow

---

## Key Components in Code

* `STFT` — Differentiable STFT module
* `STFTLoss` — Magnitude + phase loss
* `AudioDataset` — Paired clean/noisy dataset
* `AudioDiffusionModel` — Main network
* `sinusoidal_embedding()` — Diffusion-style embedding
* `train_diffusion_model()` — Training pipeline
* `denoise_audio()` — Inference pipeline

---

##  Example Workflow

1. Add clean files to `gt_audio/`
2. Run training
3. Best checkpoint saved automatically
4. Run inference on noisy file
5. Listen to reconstructed audio

---

##  License

