"""
Echo Removal — Improved with Spectral Loss
==========================================
Key fix: Combined MSE + STFT loss instead of pure MSE.
Pure MSE on waveforms causes the model to output near-silence.
Spectral loss makes the model focus on frequency content — much better for audio.

Usage:
    # Train
    python echo_removal.py --mode train --data_dir dataset/ --epochs 50 --save_model echo_model.pth

    # Infer
    python echo_removal.py --mode infer --input noisy.wav --output clean_output.wav --load_model echo_model.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE  = 16000   # 1 second
OVERLAP     = 4000


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class EchoDataset(Dataset):
    def __init__(self, data_dir, chunk_size=CHUNK_SIZE, sample_rate=SAMPLE_RATE):
        self.chunks     = []
        self.chunk_size = chunk_size

        echo_dir  = os.path.join(data_dir, "echo")
        clean_dir = os.path.join(data_dir, "clean")
        echo_files = sorted(os.listdir(echo_dir))
        print(f"Found {len(echo_files)} file(s) in dataset.")

        for fname in echo_files:
            echo_path  = os.path.join(echo_dir,  fname)
            clean_path = os.path.join(clean_dir, fname)
            if not os.path.exists(clean_path):
                continue

            echo_audio,  _ = librosa.load(echo_path,  sr=sample_rate, mono=True)
            clean_audio, _ = librosa.load(clean_path, sr=sample_rate, mono=True)

            min_len = min(len(echo_audio), len(clean_audio))
            echo_audio  = echo_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            for start in range(0, min_len - chunk_size, chunk_size // 2):
                e = echo_audio[start : start + chunk_size].astype(np.float32)
                c = clean_audio[start : start + chunk_size].astype(np.float32)

                # Normalize each chunk independently
                norm = np.max(np.abs(e)) + 1e-8
                e = e / norm
                c = c / norm

                self.chunks.append((e, c))

        print(f"Total training chunks: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        e, c = self.chunks[idx]
        return torch.tensor(e).unsqueeze(0), torch.tensor(c).unsqueeze(0)


# ─────────────────────────────────────────────
# MODEL — Deeper Conv Autoencoder with Skip Connections
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    """Simple residual block to help the model learn better features."""
    def __init__(self, channels, kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class EchoRemover(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1,  16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, 11, stride=2, padding=5), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, 9,  stride=2, padding=4), nn.BatchNorm1d(64), nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
        )

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 9,  stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, 11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(16), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1,  15, stride=2, padding=7, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        z  = self.bottleneck(e3)

        d3 = self.dec3(z)
        d3 = self._match_and_cat(d3, e2)
        d2 = self.dec2(d3)
        d2 = self._match_and_cat(d2, e1)
        d1 = self.dec1(d2)

        return d1[:, :, :x.shape[-1]]

    def _match_and_cat(self, a, b):
        min_len = min(a.shape[-1], b.shape[-1])
        return torch.cat([a[:, :, :min_len], b[:, :, :min_len]], dim=1)


# ─────────────────────────────────────────────
# SPECTRAL LOSS
# ─────────────────────────────────────────────
class SpectralLoss(nn.Module):
    """
    STFT-based loss. Compares magnitude spectrograms.
    Much better than raw waveform MSE for audio quality.
    """
    def __init__(self, fft_sizes=[256, 512, 1024]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred, target):
        pred   = pred.squeeze(1)
        target = target.squeeze(1)

        loss = 0.0
        for fft_size in self.fft_sizes:
            hop = fft_size // 4
            win = torch.hann_window(fft_size).to(pred.device)

            p_stft = torch.stft(pred,   fft_size, hop, fft_size, win, return_complex=True)
            t_stft = torch.stft(target, fft_size, hop, fft_size, win, return_complex=True)

            p_mag = torch.abs(p_stft)
            t_mag = torch.abs(t_stft)

            loss += F.l1_loss(p_mag, t_mag)
            loss += F.l1_loss(torch.log(p_mag + 1e-7), torch.log(t_mag + 1e-7))

        return loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse      = nn.MSELoss()
        self.spectral = SpectralLoss()

    def forward(self, pred, target):
        return self.mse(pred, target) + 0.5 * self.spectral(pred, target)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset    = EchoDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model     = EchoRemover().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = CombinedLoss().to(device)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for echo_batch, clean_batch in dataloader:
            echo_batch  = echo_batch.to(device)
            clean_batch = clean_batch.to(device)

            pred = model(echo_batch)
            loss = criterion(pred, clean_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        tag = " ✅ best" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_model)

        print(f"  Epoch [{epoch:3d}/{args.epochs}]  Loss: {avg_loss:.6f}{tag}")

    print(f"\nBest model saved to: {args.save_model}  (loss: {best_loss:.6f})")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EchoRemover().to(device)
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.eval()
    print(f"Loaded model from: {args.load_model}")

    audio, _ = librosa.load(args.input, sr=SAMPLE_RATE, mono=True)
    audio     = audio.astype(np.float32)
    total_len = len(audio)

    output = np.zeros(total_len, dtype=np.float32)
    count  = np.zeros(total_len, dtype=np.float32)
    step   = CHUNK_SIZE - OVERLAP

    with torch.no_grad():
        for start in range(0, total_len, step):
            end   = start + CHUNK_SIZE
            chunk = audio[start:end]

            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

            norm       = np.max(np.abs(chunk)) + 1e-8
            chunk_norm = chunk / norm

            inp  = torch.tensor(chunk_norm).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(inp).squeeze().cpu().numpy()
            pred = pred * norm

            actual_end = min(end, total_len)
            seg_len    = actual_end - start
            output[start:actual_end] += pred[:seg_len]
            count[start:actual_end]  += 1.0

    output = output / np.maximum(count, 1.0)

    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val * 0.95

    sf.write(args.output, output, SAMPLE_RATE)
    print(f"Clean audio saved to: {args.output}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       required=True, choices=["train", "infer"])
    parser.add_argument("--data_dir",   default="dataset/")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_model", default="echo_model.pth")
    parser.add_argument("--input",      default=None)
    parser.add_argument("--output",     default="output_clean.wav")
    parser.add_argument("--load_model", default="echo_model.pth")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.input:
            raise ValueError("--input is required for infer mode")
        infer(args)
