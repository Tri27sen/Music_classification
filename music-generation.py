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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class CountryMusicDataset(Dataset):
    def __init__(self, data_path, segment_length=12, sr=22050, n_mels=80, n_fft=2048):
        self.data_path = data_path
        self.segment_length = segment_length
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = 256
        
        self.country_files = []
        country_path = os.path.join(data_path, 'country')
        
        print(f"Looking for country music files in: {country_path}")
        
        if os.path.exists(country_path):
            for file in os.listdir(country_path):
                if file.endswith(('.wav', '.mp3', '.flac', '.au')):
                    file_path = os.path.join(country_path, file)
                    self.country_files.append(file_path)
            print(f"Found {len(self.country_files)} country music files")
        else:
            print(f"Country directory not found: {country_path}")
            print("Available directories:")
            if os.path.exists(data_path):
                for item in os.listdir(data_path):
                    if os.path.isdir(os.path.join(data_path, item)):
                        print(f"  - {item}")
        
        self.mel_spectrograms = []
        self.country_features = []
        self._prepare_country_data()
    
    def _prepare_country_data(self):
        print("Loading and processing country music files...")
        all_segments = []
        all_features = []
        
        target_time_frames = 1032
        
        for file_path in tqdm(self.country_files, desc="Processing country music files"):
            try:
                audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
                
                audio = librosa.effects.preemphasis(audio, coef=0.85)
                audio = librosa.util.normalize(audio)
                
                if len(audio) < self.sr * 5:
                    continue
                
                target_samples = target_time_frames * self.hop_length
                
                overlap = 0.6
                step_size = int(target_samples * (1 - overlap))
                
                for start_idx in range(0, len(audio) - target_samples, step_size):
                    end_idx = start_idx + target_samples
                    segment = audio[start_idx:end_idx]
                    
                    if np.std(segment) < 0.005:
                        continue
                    
                    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0]
                    avg_centroid = np.mean(spectral_centroid)
                    if avg_centroid < 800 or avg_centroid > 4000:
                        continue
                    
                    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
                    avg_zcr = np.mean(zero_crossing_rate)
                    if avg_zcr < 0.02 or avg_zcr > 0.3:
                        continue
                    
                    country_features = self._extract_country_features(segment)
                    
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment, 
                        sr=self.sr, 
                        n_mels=self.n_mels, 
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        fmin=80,
                        fmax=6000,
                        power=1.5
                    )
                    
                    if mel_spec.shape[1] != target_time_frames:
                        if mel_spec.shape[1] > target_time_frames:
                            mel_spec = mel_spec[:, :target_time_frames]
                        else:
                            pad_width = target_time_frames - mel_spec.shape[1]
                            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='reflect')
                    
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=60)
                    
                    mel_spec_norm = (mel_spec_db + 60) / 60
                    mel_spec_norm = 2 * mel_spec_norm - 1
                    
                    low_freq_boost = np.linspace(1.3, 0.8, self.n_mels)
                    mel_spec_norm = mel_spec_norm * low_freq_boost[:, np.newaxis]
                    
                    noise = np.random.normal(0, 0.005, mel_spec_norm.shape)
                    mel_spec_norm = mel_spec_norm + noise
                    mel_spec_norm = np.clip(mel_spec_norm, -1, 1)
                    
                    all_segments.append(mel_spec_norm)
                    all_features.append(country_features)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        self.mel_spectrograms = all_segments
        self.country_features = all_features
        print(f"Successfully loaded {len(self.mel_spectrograms)} country music mel-spectrogram segments")
        if self.mel_spectrograms:
            print(f"Each mel-spectrogram shape: {self.mel_spectrograms[0].shape}")
        else:
            print("No valid mel-spectrograms were created!")
        
        self._save_melspectrograms()
    
    def _extract_country_features(self, audio):
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sr))
        
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        country_vector = np.concatenate([
            [tempo / 200.0],
            chroma_mean,
            [spectral_centroid / 4000.0],
            [spectral_rolloff / 8000.0],
            mfcc_mean
        ])
        
        return country_vector[:27]
    
    def _save_melspectrograms(self):
        if not self.mel_spectrograms:
            print("No mel-spectrograms to save!")
            return
            
        os.makedirs('saved_melspectrograms', exist_ok=True)
        
        num_to_save = min(10, len(self.mel_spectrograms))
        
        for i in range(num_to_save):
            mel_spec = self.mel_spectrograms[i]
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(mel_spec, sr=self.sr, hop_length=self.hop_length, 
                                   x_axis='time', y_axis='mel', fmin=80, fmax=6000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Country Music Mel-Spectrogram {i+1}')
            plt.tight_layout()
            plt.savefig(f'saved_melspectrograms/country_melspec_{i+1}.png', dpi=150)
            plt.close()
        
        print(f"Saved {num_to_save} mel-spectrograms to 'saved_melspectrograms' directory")
    
    def __len__(self):
        return len(self.mel_spectrograms)
    
    def __getitem__(self, idx):
        mel_spec = self.mel_spectrograms[idx]
        features = self.country_features[idx] if self.country_features else np.zeros(27)
        return torch.FloatTensor(mel_spec), torch.FloatTensor(features)

class CountryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term * 0.8)
        pe[:, 1::2] = torch.cos(position * div_term * 0.8)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CountryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.4))
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out(attention_output)

class CountryTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.15):
        super().__init__()
        self.attention = CountryMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class CountryDiffusionBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, dropout=0.15):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.norm3 = nn.GroupNorm(min(8, channels), channels)
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x, time_emb):
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.view(time_emb.size(0), -1, 1, 1)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + time_emb
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        
        skip = self.conv3(x)
        skip = self.norm3(skip)
        
        return skip + h

class CountryTransformerDiffusionModel(nn.Module):
    def __init__(self, mel_bins=80, time_steps=800, d_model=384, n_heads=6, n_layers=6, dropout=0.15):
        super().__init__()
        self.mel_bins = mel_bins
        self.time_steps = time_steps
        self.d_model = d_model
        
        self.time_emb_dim = 192
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
        )
        
        self.country_condition = nn.Sequential(
            nn.Linear(27, 128),
            nn.SiLU(),
            nn.Linear(128, d_model),
            nn.SiLU()
        )
        
        self.input_proj = nn.Sequential(
            nn.Linear(mel_bins, d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = CountryPositionalEncoding(d_model)
        
        self.transformer_blocks = nn.ModuleList([
            CountryTransformerBlock(d_model, n_heads, d_model * 3, dropout)
            for _ in range(n_layers)
        ])
        
        self.diffusion_blocks = nn.ModuleList([
            CountryDiffusionBlock(d_model, self.time_emb_dim, dropout)
            for _ in range(3)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, mel_bins)
        )
        
        self.register_buffer('betas', self._linear_beta_schedule(time_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
        
    def get_time_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb)
    
    def forward(self, x, t, country_features=None):
        time_emb = self.get_time_embedding(t)
        
        batch_size, mel_bins, time_frames = x.size()
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        if country_features is not None:
            country_emb = self.country_condition(country_features)
            x = x + country_emb.unsqueeze(1)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = x.transpose(1, 2)
        x = x.unsqueeze(-1)
        
        for diffusion_block in self.diffusion_blocks:
            x = diffusion_block(x, time_emb)
        
        x = x.squeeze(-1)
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        x = x.transpose(1, 2)
        
        return x
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise, noise

def train_country_model(model, dataloader, num_epochs=150, learning_rate=1e-4, warmup_steps=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
    
    criterion = nn.MSELoss()
    
    print(f"Training on {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (batch, features) in enumerate(epoch_pbar):
            batch = batch.to(device)
            features = features.to(device)
            global_step += 1
            
            if global_step < warmup_steps:
                lr = learning_rate * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            t = torch.randint(0, model.time_steps, (batch.size(0),), device=device)
            
            noisy_data, noise = model.add_noise(batch, t)
            
            predicted_noise = model(noisy_data, t, features)
            
            loss = criterion(predicted_noise, noise)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            
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
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_country_model.pth')
            print(f"New best model saved with loss: {best_loss:.6f}")
        
        if (epoch + 1) % 15 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    return model

def create_country_features():
    country_features = np.array([
        110.0 / 200.0,
        0.6, 0.3, 0.8, 0.2, 0.7, 0.1, 0.5, 0.4, 0.6, 0.3, 0.8, 0.4,
        1600.0 / 4000.0,
        3200.0 / 8000.0,
        -12.2, 22.8, -7.1, 18.4, -4.5, 16.2, -2.8, 14.6, -3.9, 12.1, -2.1, 9.8, -1.7
    ])
    return country_features

def generate_country_samples(model, num_samples=15, sample_length=None, ddim_steps=50):
    device = next(model.parameters()).device
    model.eval()
    
    if sample_length is None:
        sample_length = 1032
    
    generated_samples = []
    country_features = create_country_features()
    
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Generating country music sample {i+1}/{num_samples}")
            
            x = torch.randn(1, model.mel_bins, sample_length, device=device)
            features = torch.FloatTensor(country_features).unsqueeze(0).to(device)
            
            country_emphasis = torch.linspace(1.6, 0.6, model.mel_bins, device=device)
            
            timesteps = torch.linspace(model.time_steps-1, 0, ddim_steps, dtype=torch.long, device=device)
            
            for t_idx in tqdm(range(len(timesteps)), desc="Generating"):
                t = timesteps[t_idx:t_idx+1]
                
                predicted_noise = model(x, t, features)
                
                alpha_t = model.alphas_cumprod[t]
                alpha_t_prev = model.alphas_cumprod[timesteps[t_idx+1]] if t_idx < len(timesteps)-1 else torch.tensor(1.0, device=device)
                
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                
                pred_x0 = pred_x0 * country_emphasis.view(1, -1, 1)
                
                low_freq_indices = torch.arange(0, model.mel_bins // 4, device=device)
                mid_freq_indices = torch.arange(model.mel_bins // 4, 3 * model.mel_bins // 4, device=device)
                high_freq_indices = torch.arange(3 * model.mel_bins // 4, model.mel_bins, device=device)
                
                pred_x0[:, low_freq_indices, :] *= 1.4
                pred_x0[:, mid_freq_indices, :] *= 1.2
                pred_x0[:, high_freq_indices, :] *= 0.7
                
                pred_x0 = torch.clamp(pred_x0, -1.5, 1.5)
                
                dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
                
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
                
                if t_idx > ddim_steps // 2:
                    beat_pattern = torch.sin(torch.linspace(0, 16*np.pi, sample_length, device=device)) * 0.15
                    x[:, :model.mel_bins//3, :] += beat_pattern.unsqueeze(0).unsqueeze(0)
            
            x = torch.clamp(x, -1.5, 1.5)
            generated_samples.append(x.cpu().numpy())
    
    return generated_samples

def country_melspec_to_audio(mel_spec, sr=22050, hop_length=256, n_fft=2048):
    mel_spec_db = (mel_spec + 1.5) / 3.0 * 60 - 60
    
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec_power, 
        sr=sr, 
        hop_length=hop_length, 
        n_fft=n_fft,
        fmin=80,
        fmax=6000,
        n_iter=200,
        length=None
    )
    
    audio = librosa.util.normalize(audio)
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    
    return audio

def main():
    data_path = "../input/gtzan-dataset-music-genre-classification/Data/genres_original"
    
    print("Creating country music dataset...")
    dataset = CountryMusicDataset(data_path)
    
    if len(dataset) == 0:
        print("No country music data found! Creating synthetic country data...")
        synthetic_data = []
        synthetic_features = []
        target_time_frames = 1032
        
        for i in range(200):
            time_axis = np.linspace(0, 6*np.pi, target_time_frames)
            
            country_beat = np.sin(time_axis * 1.8) * 0.5 + np.sin(time_axis * 0.9) * 0.3
            country_melody = np.sin(time_axis * 0.6) * 0.4 + np.cos(time_axis * 1.3) * 0.3
            country_bass = np.sin(time_axis * 0.4) * 0.7 + np.cos(time_axis * 0.2) * 0.4
            
            country_twang = np.sin(time_axis * 2.2) * 0.2 + np.random.randn(target_time_frames) * 0.08
            
            freq_patterns = np.random.randn(80, 1) * 0.05
            
            country_emphasis = np.array([1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9] + [0.8] * 12 + [0.7] * 20 + [0.6] * 40)
            
            mel_spec = (freq_patterns + country_beat + country_melody + country_bass + country_twang) * country_emphasis[:, np.newaxis]
            
            low_freq_boost = np.zeros_like(mel_spec)
            low_freq_boost[:15, :] = 0.4
            low_freq_boost[15:30, :] = 0.3
            low_freq_boost[30:50, :] = 0.2
            
            mel_spec += low_freq_boost
            
            beat_emphasis = np.tile(np.array([1.2, 1.0, 1.1, 1.0]), target_time_frames // 4 + 1)[:target_time_frames]
            mel_spec[:20, :] *= beat_emphasis
            
            mel_spec = np.tanh(mel_spec)
            
            country_features = create_country_features()
            country_features += np.random.normal(0, 0.08, country_features.shape)
            
            synthetic_data.append(mel_spec)
            synthetic_features.append(country_features)
        
        dataset.mel_spectrograms = synthetic_data
        dataset.country_features = synthetic_features
        print(f"Created {len(synthetic_data)} synthetic country music mel-spectrograms")
        dataset.mel_spectrograms = synthetic_data
        dataset.country_features = synthetic_features
        print(f"Created {len(synthetic_data)} synthetic country music mel-spectrograms")
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2, pin_memory=True)
    else:
        print("ERROR: No data available for training!")
        return
    
    print("Creating country music transformer diffusion model...")
    model = CountryTransformerDiffusionModel(
        mel_bins=80, 
        time_steps=800,
        d_model=384,
        n_heads=6,
        n_layers=6,
        dropout=0.15
    )
    
    print("Training country music model...")
    trained_model = train_country_model(model, dataloader, num_epochs=150, learning_rate=1.5e-4)
    
    print("Saving country music model...")
    
    torch.save(trained_model.state_dict(), 'country_music_model.pth')
    print("Model saved as 'country_music_model.pth'")
    

    print("Generating country music samples...")
    generated_samples = generate_country_samples(trained_model, num_samples=25, ddim_steps=40)
    
    os.makedirs('generated_country_music', exist_ok=True)
    for i, sample in enumerate(generated_samples):
        mel_spec = sample[0]
        audio = country_melspec_to_audio(mel_spec)
        
        output_path = f'generated_country_music/country_sample_{i+1}.wav'
        torchaudio.save(output_path, torch.FloatTensor(audio).unsqueeze(0), 22050)
        print(f"Saved: {output_path}")
        
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(mel_spec, sr=22050, hop_length=256, 
                               x_axis='time', y_axis='mel', fmin=80, fmax=6000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Generated Country Music Mel-Spectrogram {i+1}')
        plt.tight_layout()
        plt.savefig(f'generated_country_music/country_melspec_{i+1}.png', dpi=150)
        plt.close()
        
    
    print(f"Generated {len(generated_samples)} country music samples!")
    print("Files saved in 'generated_country_music' directory")
    
    print("Country music generation complete!")


if __name__ == "__main__":
    main()
