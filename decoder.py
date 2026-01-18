"""
Decoder module for Soprano - converts LLM hidden states to waveforms.
This is a simplified decoder that can be trained jointly with the LLM.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ISTFTHead(nn.Module):
    """
    Inverse STFT head for waveform generation.
    Based on Vocos architecture.
    """
    def __init__(
        self,
        dim: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        padding: str = "same",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] - hidden states from LLM
        Returns:
            audio: [B, 1, L] - waveform
        """
        x = self.out(x)  # [B, T, n_fft + 2]
        x = x.transpose(1, 2)  # [B, n_fft + 2, T]
        
        mag, phase = x.chunk(2, dim=1)  # each [B, (n_fft+2)//2, T]
        mag = torch.exp(mag)
        # Note: Using sin() for phase is a simplification. More sophisticated methods
        # like Griffin-Lim or learned phase prediction could improve quality.
        phase = torch.sin(phase)
        
        # Construct complex spectrogram
        S = mag * torch.exp(1j * phase)
        
        # Inverse STFT
        audio = torch.istft(
            S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            center=True,
        )
        
        return audio.unsqueeze(1)  # [B, 1, L]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for 1D processing."""
    
    def __init__(self, dim: int, intermediate_dim: int, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = residual + x
        return x


class SopranoDecoder(nn.Module):
    """
    Decoder for Soprano that converts LLM hidden states to audio waveforms.
    This is trained jointly with the LLM to align hidden states with audio.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # LLM hidden dimension
        hidden_dim: int = 512,
        num_layers: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length
        
        # Project LLM hidden states to decoder hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # ConvNeXt layers for processing
        self.layers = nn.ModuleList([
            ConvNeXtBlock(
                dim=hidden_dim,
                intermediate_dim=hidden_dim * 4,
                kernel_size=7,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # ISTFT head for waveform generation
        self.head = ISTFTHead(
            dim=hidden_dim,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] - LLM hidden states for audio tokens
        Returns:
            waveform: [B, 1, L] - generated audio waveform
        """
        # Project to decoder dimension
        x = self.input_proj(hidden_states)  # [B, T, hidden_dim]
        
        # Transpose for conv layers
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        
        # Process through ConvNeXt layers
        for layer in self.layers:
            x = layer(x)
        
        # Transpose back for layer norm
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        x = self.norm(x)
        
        # Generate waveform
        waveform = self.head(x)  # [B, 1, L]
        
        return waveform
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        """Load decoder from checkpoint."""
        import os
        decoder_path = os.path.join(path, "decoder.pth")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder checkpoint not found at {decoder_path}")
        
        state_dict = torch.load(decoder_path, map_location=device)
        
        # Infer dimensions from state dict
        input_dim = state_dict["input_proj.weight"].shape[1]
        hidden_dim = state_dict["input_proj.weight"].shape[0]
        num_layers = sum(1 for k in state_dict.keys() if "layers." in k and ".dwconv.weight" in k)
        
        decoder = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        decoder.load_state_dict(state_dict)
        return decoder
