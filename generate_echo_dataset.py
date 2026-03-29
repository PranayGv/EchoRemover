"""
Echo Dataset Generator
======================
Takes a folder of clean .wav files and generates echo versions
using pyroomacoustics room simulation.

Output folder structure (ready for echo_removal.py):
    dataset/
        clean/   ← copies of original files
        echo/    ← simulated echo versions (same filenames)

Usage:
    python generate_echo_dataset.py --input_dir cmu_us_bdl_arctic/wav/ --output_dir dataset/ --limit 300

Install dependencies:
    pip install pyroomacoustics soundfile librosa numpy
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import pyroomacoustics as pra

# ─────────────────────────────────────
# CONFIG — tweak these to change echo strength
# ─────────────────────────────────────
SAMPLE_RATE = 16000

# Room dimensions [length, width, height] in meters
ROOM_DIM = [6, 5, 3]

# Source position [x, y, z]
SOURCE_POS = [2, 2, 1.5]

# Mic position [x, y, z]
MIC_POS = [4, 3, 1.5]

# How much echo: lower absorption = more echo, higher = less echo
# Range: 0.1 (very strong echo) to 0.9 (almost no echo)
ABSORPTION = 0.25

# Max reflections — higher = more realistic room echo
MAX_ORDER = 8


def add_echo(clean_audio, sr):
    """Simulate a room echo using pyroomacoustics."""
    room = pra.ShoeBox(
        ROOM_DIM,
        fs=sr,
        max_order=MAX_ORDER,
        absorption=ABSORPTION
    )

    room.add_source(SOURCE_POS, signal=clean_audio)

    mic_array = np.array(MIC_POS).reshape(3, 1)  # shape (3, 1)
    room.add_microphone(mic_array)

    room.simulate()

    echo_audio = room.mic_array.signals[0]

    # Match length to original
    echo_audio = echo_audio[:len(clean_audio)]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(echo_audio))
    if max_val > 0:
        echo_audio = echo_audio / max_val * 0.9

    return echo_audio.astype(np.float32)


def main(args):
    clean_out = os.path.join(args.output_dir, "clean")
    echo_out  = os.path.join(args.output_dir, "echo")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(echo_out,  exist_ok=True)

    # Collect all wav files
    wav_files = [f for f in os.listdir(args.input_dir) if f.endswith(".wav")]
    wav_files = sorted(wav_files)

    if args.limit:
        wav_files = wav_files[:args.limit]

    print(f"Processing {len(wav_files)} files...")
    print(f"Echo settings: room={ROOM_DIM}m, absorption={ABSORPTION}, max_order={MAX_ORDER}\n")

    for i, fname in enumerate(wav_files):
        input_path = os.path.join(args.input_dir, fname)

        # Load and resample
        audio, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        # Generate echo version
        echo_audio = add_echo(audio, SAMPLE_RATE)

        # Save both with the same filename
        sf.write(os.path.join(clean_out, fname), audio,      SAMPLE_RATE)
        sf.write(os.path.join(echo_out,  fname), echo_audio, SAMPLE_RATE)

        print(f"  [{i+1:3d}/{len(wav_files)}] {fname}")

    print(f"\nDone! Dataset saved to: {args.output_dir}")
    print(f"  clean/ → {len(wav_files)} files")
    print(f"  echo/  → {len(wav_files)} files")
    print(f"\nNow train with:")
    print(f"  python echo_removal.py --mode train --data_dir {args.output_dir} --epochs 20")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate echo dataset from clean audio files")
    parser.add_argument("--input_dir",  required=True,       help="Folder with clean .wav files")
    parser.add_argument("--output_dir", default="dataset/",  help="Output folder (creates clean/ and echo/ inside)")
    parser.add_argument("--limit",      type=int, default=None, help="Max number of files to process (default: all)")
    args = parser.parse_args()
    main(args)
