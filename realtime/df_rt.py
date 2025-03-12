import numpy as np
import soundfile as sf
import resampy
from deep_filter_rt import RealtimeDf
import argparse
import time


def process_audio(input_file, output_file, atten_lim):
    # Load the audio file
    audio, sr = sf.read(input_file)
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Create deep filter instance (mono)
    df = RealtimeDf(channels=1, atten_lim=atten_lim)
    target_sr = df.sample_rate

    # Resample to target sample rate if needed
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz")
        audio = resampy.resample(audio, sr, target_sr)

    # Process audio in chunks of hop_size
    hop_size = df.hop_size
    n_chunks = len(audio) // hop_size

    # Pre-allocate output buffer
    output = np.zeros_like(audio)

    print(f"Processing {n_chunks} chunks...")
    processing_start_time = time.time()

    for i in range(n_chunks):
        start_idx = i * hop_size
        end_idx = start_idx + hop_size

        # Prepare input chunk (shape: channels x hop_size)
        chunk = audio[start_idx:end_idx]
        chunk = chunk.reshape(1, -1)  # Add channel dimension

        # Process through deep filter
        enhanced = df.process_frames(chunk)

        # Store in output buffer
        output[start_idx:end_idx] = enhanced[0]  # Remove channel dimension

    # Handle remaining samples if any
    remaining = len(audio) % hop_size
    if remaining > 0:
        start_idx = n_chunks * hop_size
        last_chunk = np.zeros((1, hop_size), dtype=np.float32)
        last_chunk[0, :remaining] = audio[start_idx:]
        enhanced = df.process_frames(last_chunk)
        output[start_idx : start_idx + remaining] = enhanced[0, :remaining]

    processing_time = time.time() - processing_start_time
    rtf = processing_time / audio_duration
    processing_speed = audio_duration / processing_time

    print(f"\nProcessing Statistics:")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Real-time factor (RTF): {rtf:.2f}x")
    print(f"Processing speed: {processing_speed:.2f}x real-time")

    # Resample back to original sample rate if needed
    if sr != target_sr:
        print(f"Resampling back to {sr}Hz")
        output = resampy.resample(output, target_sr, sr)

    # Save output
    sf.write(output_file, output, sr)
    print(f"Enhanced audio saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process audio through DeepFilter")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("-a", "--atten-lim", type=float, default=100., help="Attenuation limit (0-100)")
    args = parser.parse_args()

    process_audio(args.input, args.output, args.atten_lim)


if __name__ == "__main__":
    main()
