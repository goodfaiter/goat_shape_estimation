import torch
import math
from abc import ABC, abstractmethod


class Noise(ABC):
    """Base class for noise generation."""

    def __init__(self, index = None, **kwargs):
        super().__init__()
        self._index = index
        self._validate_params(**kwargs)
        self._set_params(**kwargs)

    @abstractmethod
    def _validate_params(self, **kwargs):
        """Validate noise parameters."""
        pass

    @abstractmethod
    def _set_params(self, **kwargs):
        """Set noise parameters."""
        pass

    @abstractmethod
    def _generate_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate noise with the same shape as input tensor."""
        pass

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise to input tensor without modifying the original.

        Args:
            tensor: Input tensor of shape [T, D]

        Returns:
            Noisy tensor of same shape [T, D]
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if tensor.dim() != 2:
            raise ValueError("Input tensor must be 2D with shape [T, D]")

        noisy_tensor = torch.empty_like(tensor).copy_(tensor)
        
        if self._index is None:
            return noisy_tensor
        
        noisy_tensor[:, self._index] += self._generate_noise(tensor)
        return noisy_tensor


class NoiseGaussian(Noise):
    """Additive Gaussian noise."""

    def _validate_params(self, mean=0.0, std=1.0):
        if std <= 0:
            raise ValueError("Standard deviation must be positive")

    def _set_params(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def _generate_noise(self, tensor):
        return torch.randn_like(tensor) * self.std + self.mean


class NoiseOffset(Noise):
    """Constant offset noise."""

    def _validate_params(self, offset=0.1):
        if offset <= 0:
            raise ValueError("Offset has to be positive")

    def _set_params(self, offset=0.1):
        self.offset = offset

    def _generate_noise(self, tensor):
        min_segment_length = 20
        max_segment_length = 100
        device = tensor.device
        T = tensor.shape[0]
        D = len(self._index)

        # Estimate number of segments needed
        avg_segment_length = (min_segment_length + max_segment_length) / 2
        est_segments = int(T / avg_segment_length) + 2
        
        # Generate segment lengths
        segment_lengths = torch.randint(
            min_segment_length, max_segment_length + 1,
            (est_segments,), device=device
        )
        
        # Create cumulative lengths and find where we exceed T
        cum_lengths = torch.cumsum(segment_lengths, dim=0)
        valid_segments = torch.where(cum_lengths <= T)[0]
        
        if len(valid_segments) == 0:
            # Handle case where first segment is longer than T
            segment_lengths = torch.tensor([T], device=device)
            num_segments = 1
        else:
            num_segments = len(valid_segments)
            segment_lengths = segment_lengths[:num_segments]
            segment_lengths[-1] = T - (0 if num_segments == 1 else cum_lengths[num_segments-2])
        
        # Generate offsets for each segment
        offsets = torch.empty((num_segments, D), device=device).uniform_(-self.offset, self.offset)
        
        # Create the final tensor using repeat_interleave
        noise = torch.repeat_interleave(offsets, segment_lengths[:num_segments], dim=0)

        return noise


        # dist = torch.distributions.Uniform(low=-1.0 * self.offset, high=self.offset)
        # # Note we want different offsets in every dimensions but same over the time series
        # return torch.ones_like(tensor[:, self._index]) * dist.sample([tensor.shape[0], len(self._index)]).to(tensor.device)


class NoiseSinusoidal(Noise):
    """Sinusoidal noise."""

    def _validate_params(self, amplitude=0.1, frequency=1.0):
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive")
        if frequency <= 0:
            raise ValueError("Frequency must be positive")

    def _set_params(self, amplitude=0.1, frequency=1.0):
        self.amplitude = amplitude
        self.frequency = frequency

    def _generate_noise(self, tensor):
        T, D = tensor.shape
        time_steps = torch.arange(T, dtype=torch.float32, device=tensor.device)
        # Shape [T, 1] to broadcast across D dimensions
        noise = self.amplitude * torch.sin(2 * math.pi * self.frequency * time_steps)
        return noise.unsqueeze(-1).expand(-1, D)


class NoiseClean(Noise):
    """No noise (returns clean tensor)."""

    def _validate_params(self):
        pass

    def _set_params(self):
        pass

    def _generate_noise(self, tensor):
        return torch.zeros_like(tensor)


# Example usage:
if __name__ == "__main__":
    T, D = 10, 3  # 10 timesteps, 3 dimensions
    original_tensor = torch.zeros(T, D)

    # Create different noise types
    gaussian_noise = NoiseGaussian(mean=0.0, std=0.1)
    offset_noise = NoiseOffset(offset=0.2)
    sin_noise = NoiseSinusoidal(amplitude=0.3, frequency=0.5)
    clean_noise = NoiseClean()

    # Apply noises
    noisy_gaussian = gaussian_noise(original_tensor)
    noisy_offset = offset_noise(original_tensor)
    noisy_sin = sin_noise(original_tensor)
    noisy_clean = clean_noise(original_tensor)

    print("Original tensor remains unchanged:", original_tensor)
    print("Gaussian noise example:", noisy_gaussian)
    print("Offset noise example:", noisy_offset)
    print("Sinusoidal noise example:", noisy_sin)
    print("Clean noise example:", noisy_clean)
