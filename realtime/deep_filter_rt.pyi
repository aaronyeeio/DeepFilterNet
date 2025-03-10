from typing import Tuple
import numpy as np
import numpy.typing as npt

class RealtimeDf:
    """Real-time DeepFilter noise suppression processor.
    
    This class provides real-time audio processing capabilities using the DeepFilter
    noise suppression algorithm.
    """
    
    def __init__(self, channels: int) -> None:
        """Initialize a new RealtimeDf instance.
        
        Args:
            channels: Number of audio channels (1 for mono, 2 for stereo)
        
        Raises:
            RuntimeError: If initialization fails
        """
        ...
    
    def process_frames(self, input_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Process a chunk of audio frames.
        
        Args:
            input_array: Input audio array of shape (channels, samples)
                        where samples must equal hop_size
        
        Returns:
            Processed audio array of shape (channels, samples)
        
        Raises:
            RuntimeError: If processing fails
        """
        ...
    
    @property
    def hop_size(self) -> int:
        """Get the hop size (number of samples per frame)."""
        ...
    
    @property
    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        ...
    
    @property
    def channels(self) -> int:
        """Get the number of audio channels."""
        ... 