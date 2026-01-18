"""
Audio waveform losses for decoder training.
Includes mel-spectrogram loss and multi-resolution STFT loss.
"""
import torch
import torchaudio


def stft_loss(x, y, fft_sizes=(1024, 2048, 512), hop_sizes=(256, 512, 128), win_lengths=(1024, 2048, 512)):
    """
    Multi-resolution STFT loss.
    
    Args:
        x: predicted waveform [B, 1, L] or [B, L]
        y: ground truth waveform [B, 1, L] or [B, L]
        fft_sizes: tuple of FFT sizes
        hop_sizes: tuple of hop lengths
        win_lengths: tuple of window lengths
    
    Returns:
        loss: scalar tensor
    """
    # Ensure inputs are 2D [B, L]
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    
    loss = 0.0
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        # Compute STFT
        window = torch.hann_window(win).to(x.device)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, 
                       window=window, return_complex=True, center=True)
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, 
                       window=window, return_complex=True, center=True)
        
        # Magnitude loss (L1)
        loss = loss + (X.abs() - Y.abs()).abs().mean()
        
    return loss / len(fft_sizes)


def mel_loss(x, y, sr=32000, n_mels=128, n_fft=1024, hop_length=256):
    """
    Mel-spectrogram loss.
    
    Args:
        x: predicted waveform [B, 1, L] or [B, L]
        y: ground truth waveform [B, 1, L] or [B, L]
        sr: sample rate
        n_mels: number of mel bins
        n_fft: FFT size
        hop_length: hop length
    
    Returns:
        loss: scalar tensor
    """
    # Ensure inputs are 2D [B, L]
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    
    # Create mel transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
    ).to(x.device)
    
    # Compute mel spectrograms
    X = mel_transform(x)
    Y = mel_transform(y)
    
    # Log mel spectrograms (add small epsilon for stability)
    X = torch.log(X + 1e-5)
    Y = torch.log(Y + 1e-5)
    
    # L1 loss
    return (X - Y).abs().mean()


def audio_decoder_loss(pred_wav, gt_wav, sr=32000):
    """
    Combined audio loss for decoder training.
    
    Args:
        pred_wav: predicted waveform [B, 1, L] or [B, L]
        gt_wav: ground truth waveform [B, 1, L] or [B, L]
        sr: sample rate
    
    Returns:
        loss: scalar tensor
        loss_dict: dict with individual loss components
    """
    mel_l = mel_loss(pred_wav, gt_wav, sr=sr)
    stft_l = stft_loss(pred_wav, gt_wav)
    
    # Combine losses
    total_loss = mel_l + stft_l
    
    loss_dict = {
        "mel_loss": mel_l.item(),
        "stft_loss": stft_l.item(),
        "total_audio_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict
