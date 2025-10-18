import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import random
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ImprovedMelSpectrogramDataset(Dataset):
    def __init__(self, data_path, segment_length=8, sr=22050, n_mels=128, n_fft=2048):
        self.data_path = data_path
        self.segment_length = segment_length
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = 512
        
        # Load classical music files
        self.classical_files = []
        classical_path = os.path.join(data_path, 'classical')
        if os.path.exists(classical_path):
            for file in os.listdir(classical_path):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    self.classical_files.append(os.path.join(classical_path, file))
        
        self.mel_spectrograms = []
        self.scaler = StandardScaler()
        self._prepare_data()
    
    def _prepare_data(self):
        print("Loading and processing classical music files...")
        all_segments = []
        
        # Target dimensions for consistency (longer segments for better quality)
        target_time_frames = 644  # ~15 seconds at 22050 Hz with hop_length=512
        
        for file_path in tqdm(self.classical_files):
            try:
                # Load audio file with better preprocessing
                audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
                
                # Apply audio preprocessing
                audio = librosa.effects.preemphasis(audio)
                audio = librosa.util.normalize(audio)
                
                # Calculate exact segment length in samples
                target_samples = target_time_frames * self.hop_length
                
                # Use overlapping windows for more data
                overlap = 0.5
                step_size = int(target_samples * (1 - overlap))
                
                # Extract overlapping segments
                for start_idx in range(0, len(audio) - target_samples, step_size):
                    end_idx = start_idx + target_samples
                    segment = audio[start_idx:end_idx]
                    
                    # Skip segments that are too quiet
                    if np.std(segment) < 0.001:
                        continue
                    
                    # Convert to mel-spectrogram with improved parameters
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment, 
                        sr=self.sr, 
                        n_mels=self.n_mels, 
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        fmin=80,  # Focus on musical frequencies
                        fmax=8000,
                        power=2.0
                    )
                    
                    # Ensure consistent time dimension
                    if mel_spec.shape[1] != target_time_frames:
                        if mel_spec.shape[1] > target_time_frames:
                            mel_spec = mel_spec[:, :target_time_frames]
                        else:
                            pad_width = target_time_frames - mel_spec.shape[1]
                            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='edge')
                    
                    # Convert to dB scale with better normalization
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
                    
                    # Normalize to [-1, 1] with improved method
                    mel_spec_norm = (mel_spec_db + 80) / 80  # [0, 1]
                    mel_spec_norm = 2 * mel_spec_norm - 1   # [-1, 1]
                    
                    # Add slight noise for regularization
                    noise = np.random.normal(0, 0.01, mel_spec_norm.shape)
                    mel_spec_norm = mel_spec_norm + noise
                    mel_spec_norm = np.clip(mel_spec_norm, -1, 1)
                    
                    all_segments.append(mel_spec_norm)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        self.mel_spectrograms = all_segments
        print(f"Loaded {len(self.mel_spectrograms)} mel-spectrogram segments")
        print(f"Each mel-spectrogram shape: {self.mel_spectrograms[0].shape if self.mel_spectrograms else 'No data'}")
        
        # Save mel-spectrograms
        self._save_melspectrograms()
    
    def _save_melspectrograms(self):
        os.makedirs('saved_melspectrograms', exist_ok=True)
        
        for i, mel_spec in enumerate(self.mel_spectrograms[:10]):
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(mel_spec, sr=self.sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', fmin=80, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Classical Music Mel-Spectrogram {i+1}')
            plt.tight_layout()
            plt.savefig(f'saved_melspectrograms/classical_melspec_{i+1}.png', dpi=150)
            plt.close()
        
        print("Saved mel-spectrograms to 'saved_melspectrograms' directory")
    
    def __len__(self):
        return len(self.mel_spectrograms)
    
    def __getitem__(self, idx):
        mel_spec = self.mel_spectrograms[idx]
        return torch.FloatTensor(mel_spec)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImprovedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention with learnable temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out(attention_output)

class ImprovedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = ImprovedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Improved feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),  # GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Pre-normalization (more stable training)
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class ImprovedDiffusionBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
        # Improved convolution layers
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 1)  # 1x1 conv for skip connection
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.norm3 = nn.GroupNorm(min(8, channels), channels)
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x, time_emb):
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.view(time_emb.size(0), -1, 1, 1)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = h + time_emb
        h = self.dropout(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Skip connection with 1x1 conv
        skip = self.conv3(x)
        skip = self.norm3(skip)
        
        return skip + h

class ImprovedTransformerDiffusionModel(nn.Module):
    def __init__(self, mel_bins=128, time_steps=1000, d_model=512, n_heads=8, n_layers=8, dropout=0.1):
        super().__init__()
        self.mel_bins = mel_bins
        self.time_steps = time_steps
        self.d_model = d_model
        
        # Improved time embedding
        self.time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
        )
        
        # Input projection with residual connection
        self.input_proj = nn.Sequential(
            nn.Linear(mel_bins, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Improved transformer layers
        self.transformer_blocks = nn.ModuleList([
            ImprovedTransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Improved diffusion blocks
        self.diffusion_blocks = nn.ModuleList([
            ImprovedDiffusionBlock(d_model, self.time_emb_dim, dropout)
            for _ in range(4)  # More diffusion blocks
        ])
        
        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, mel_bins)
        )
        
        # Improved noise schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(time_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def get_time_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb)
    
    def forward(self, x, t):
        # Get time embeddings
        time_emb = self.get_time_embedding(t)
        
        # Input projection
        batch_size, mel_bins, time_frames = x.size()
        x = x.transpose(1, 2)  # (batch, time_frames, mel_bins)
        x = self.input_proj(x)  # (batch, time_frames, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (time_frames, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, time_frames, d_model)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Reshape for diffusion blocks
        x = x.transpose(1, 2)  # (batch, d_model, time_frames)
        x = x.unsqueeze(-1)  # (batch, d_model, time_frames, 1)
        
        # Diffusion blocks
        for diffusion_block in self.diffusion_blocks:
            x = diffusion_block(x, time_emb)
        
        # Output projection
        x = x.squeeze(-1)  # (batch, d_model, time_frames)
        x = x.transpose(1, 2)  # (batch, time_frames, d_model)
        x = self.output_proj(x)  # (batch, time_frames, mel_bins)
        x = x.transpose(1, 2)  # (batch, mel_bins, time_frames)
        
        return x
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise, noise

def train_improved_model(model, dataloader, num_epochs=100, learning_rate=1e-4, warmup_steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Improved optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
    
    # Loss function with label smoothing
    criterion = nn.SmoothL1Loss()  # Huber loss for more robust training
    
    print(f"Training on {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(epoch_pbar):
            batch = batch.to(device)
            global_step += 1
            
            # Warmup learning rate
            if global_step < warmup_steps:
                lr = learning_rate * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Random timesteps with importance sampling
            t = torch.randint(0, model.time_steps, (batch.size(0),), device=device)
            
            # Add noise
            noisy_data, noise = model.add_noise(batch, t)
            
            # Predict noise
            predicted_noise = model(noisy_data, t)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step >= warmup_steps:
                scheduler.step()
            
            total_loss += loss.item()
            epoch_pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_transformer_diffusion_model.pth')
            print(f"New best model saved with loss: {best_loss:.6f}")
    
    return model

def generate_improved_samples(model, num_samples=8, sample_length=None, use_ddim=True, ddim_steps=50):
    device = next(model.parameters()).device
    model.eval()
    
    if sample_length is None:
        sample_length = 644  # Longer samples for better quality
    
    generated_samples = []
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}")
            
            # Start with pure noise
            x = torch.randn(1, model.mel_bins, sample_length, device=device)
            
            if use_ddim:
                # DDIM sampling for faster and better quality generation
                timesteps = torch.linspace(model.time_steps-1, 0, ddim_steps, dtype=torch.long, device=device)
                
                for t_idx in tqdm(range(len(timesteps)), desc="DDIM Sampling"):
                    t = timesteps[t_idx:t_idx+1]
                    
                    # Predict noise
                    predicted_noise = model(x, t)
                    
                    # DDIM update
                    alpha_t = model.alphas_cumprod[t]
                    alpha_t_prev = model.alphas_cumprod[timesteps[t_idx+1]] if t_idx < len(timesteps)-1 else torch.tensor(1.0, device=device)
                    
                    # Predicted x0
                    pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                    pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clamp to valid range
                    
                    # Direction pointing to xt
                    dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
                    
                    # Update x
                    x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            else:
                # Standard DDPM sampling
                for t in tqdm(reversed(range(model.time_steps)), desc="DDPM Sampling"):
                    t_tensor = torch.tensor([t], device=device)
                    
                    # Predict noise
                    predicted_noise = model(x, t_tensor)
                    
                    # Remove noise
                    alpha = model.alphas[t]
                    alpha_cumprod = model.alphas_cumprod[t]
                    beta = model.betas[t]
                    
                    if t > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = 0
                    
                    x = (1 / torch.sqrt(alpha)) * (x - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod)) + torch.sqrt(beta) * noise
            
            # Post-process generated sample
            x = torch.clamp(x, -1, 1)
            generated_samples.append(x.cpu().numpy())
    
    return generated_samples

def improved_melspec_to_audio(mel_spec, sr=22050, hop_length=512, n_fft=2048):
    """Improved mel-spectrogram to audio conversion with better quality"""
    # Convert from [-1, 1] to dB scale
    mel_spec_db = (mel_spec + 1) / 2 * 80 - 80  # [-80, 0] dB
    
    # Convert dB to power
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    
    # Improved inverse mel-spectrogram with Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec_power, 
        sr=sr, 
        hop_length=hop_length, 
        n_fft=n_fft,
        fmin=80,
        fmax=8000,
        n_iter=64,  # More iterations for better quality
        length=None
    )
    
    # Post-process audio
    audio = librosa.util.normalize(audio)
    
    return audio

def main():
    # Configuration
    data_path = "../input/gtzan-dataset-music-genre-classification/Data/genres_original"
    
    # Create dataset
    print("Creating improved dataset...")
    dataset = ImprovedMelSpectrogramDataset(data_path)
    
    if len(dataset) == 0:
        print("No data found! Creating synthetic data for demonstration...")
        # Create more realistic synthetic data
        synthetic_data = []
        target_time_frames = 644
        for i in range(100):  # More samples
            # Create more realistic mel-spectrogram patterns
            time_axis = np.linspace(0, 4*np.pi, target_time_frames)
            freq_patterns = np.random.randn(128, 1) * 0.1
            time_patterns = np.sin(time_axis) * 0.3 + np.cos(time_axis * 0.5) * 0.2
            mel_spec = freq_patterns + time_patterns
            mel_spec = np.tanh(mel_spec)  # Smooth activation
            synthetic_data.append(mel_spec)
        dataset.mel_spectrograms = synthetic_data
        print(f"Created {len(synthetic_data)} synthetic mel-spectrograms with shape: {synthetic_data[0].shape}")
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    
    # Create improved model
    print("Creating improved model...")
    model = ImprovedTransformerDiffusionModel(
        mel_bins=128, 
        time_steps=1000,  # More time steps for better quality
        d_model=512,
        n_heads=8,
        n_layers=8,  # More layers
        dropout=0.1
    )
    
    # Train model
    print("Training improved model...")
    trained_model = train_improved_model(model, dataloader, num_epochs=100, learning_rate=2e-4)
    
    # Save model
    print("Saving model...")
    torch.save(trained_model.state_dict(), 'improved_transformer_diffusion_model.pth')
    print("Model saved as 'improved_transformer_diffusion_model.pth'")
    
    # Generate samples
    print("Generating improved audio samples...")
    generated_samples = generate_improved_samples(trained_model, num_samples=8, use_ddim=True, ddim_steps=50)
    
    # Convert to audio and save
    os.makedirs('generated_audio_improved', exist_ok=True)
    for i, sample in enumerate(generated_samples):
        mel_spec = sample[0]  # Remove batch dimension
        audio = improved_melspec_to_audio(mel_spec)
        
        # Save audio
        output_path = f'generated_audio_improved/classical_sample_{i+1}.wav'
        torchaudio.save(output_path, torch.FloatTensor(audio).unsqueeze(0), 22050)
        print(f"Saved {output_path}")
    
    print("Improved generation complete!")

if __name__ == "__main__":
    main()
