import numpy as np
import time
from deep_filter_rt import RealtimeDf

def main():
    # Create a stereo (2 channel) instance
    df = RealtimeDf(channels=2, atten_lim=100.)
    
    # Create some dummy input data
    # Shape must be (channels, hop_size)
    dummy_input = np.zeros((2, df.hop_size), dtype=np.float32)
    
    # Process the frames
    start = time.time()
    output = df.process_frames(dummy_input)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    print(f"Processed frame shape: {output.shape}")
    print(f"Sample rate: {df.sample_rate} Hz")
    print(f"Hop size: {df.hop_size} samples")
    print(f"Channels: {df.channels}")

if __name__ == "__main__":
    main()