<<<<<<< HEAD
import numpy as np
import soundfile as sf
import scipy.signal
import os
import glob

def spectral_subtract_scipy(noisy, sr, n_fft=256, hop_length=128, win_length=256,
                             alpha=4.0, beta=1e-4):
    """
    使用 SciPy 的 STFT 实现谱减，避免 librosa/numba 依赖
    """
    f, t, Zxx = scipy.signal.stft(noisy, fs=sr, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    power = mag**2

    # 估计噪声功率
    noise_mag = np.mean(mag[:, :30], axis=1, keepdims=True)
    noise_pow = np.tile(noise_mag**2, (1, power.shape[1]))

    # 谱减
    gamma = 1.0
    P_enh = np.power(power, gamma) - alpha * np.power(noise_pow, gamma)
    P_enh = np.maximum(P_enh, 0.0)**(1.0 / gamma)

    # 阈值处理
    mask = P_enh >= beta * noise_pow
    P_enh = P_enh * mask + beta * noise_pow * (~mask)

    mag_enh = np.sqrt(P_enh)
    Zxx_enh = mag_enh * np.exp(1j * phase)

    # iSTFT
    _, y_enh = scipy.signal.istft(Zxx_enh, fs=sr, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft)
    return y_enh


# 处理音频文件（替代原始的 librosa 方案）
INPUT_ROOT = "train_audio"
OUTPUT_ROOT = "train_audio_clear"  # Kaggle 的写入目录

pattern = os.path.join(INPUT_ROOT, "*", "*.ogg")
all_files = glob.glob(pattern)

for src_path in all_files:
    # 读取音频
    y, sr = sf.read(src_path)  # 用 soundfile 读 .ogg
    if y.ndim > 1:
        y = y[:, 0]  # 转为单通道

    # 增强
    y_enh = spectral_subtract_scipy(y, sr)

    # 保存
    rel = os.path.relpath(src_path, INPUT_ROOT)
    dst_path = os.path.join(OUTPUT_ROOT, rel).replace(".ogg", ".flac")
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, y_enh, sr)
=======
import numpy as np
import soundfile as sf
import scipy.signal
import os
import glob

def spectral_subtract_scipy(noisy, sr, n_fft=256, hop_length=128, win_length=256,
                             alpha=4.0, beta=1e-4):
    """
    使用 SciPy 的 STFT 实现谱减，避免 librosa/numba 依赖
    """
    f, t, Zxx = scipy.signal.stft(noisy, fs=sr, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    power = mag**2

    # 估计噪声功率
    noise_mag = np.mean(mag[:, :30], axis=1, keepdims=True)
    noise_pow = np.tile(noise_mag**2, (1, power.shape[1]))

    # 谱减
    gamma = 1.0
    P_enh = np.power(power, gamma) - alpha * np.power(noise_pow, gamma)
    P_enh = np.maximum(P_enh, 0.0)**(1.0 / gamma)

    # 阈值处理
    mask = P_enh >= beta * noise_pow
    P_enh = P_enh * mask + beta * noise_pow * (~mask)

    mag_enh = np.sqrt(P_enh)
    Zxx_enh = mag_enh * np.exp(1j * phase)

    # iSTFT
    _, y_enh = scipy.signal.istft(Zxx_enh, fs=sr, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft)
    return y_enh


# 处理音频文件（替代原始的 librosa 方案）
INPUT_ROOT = "train_audio"
OUTPUT_ROOT = "train_audio_clear"  # Kaggle 的写入目录

pattern = os.path.join(INPUT_ROOT, "*", "*.ogg")
all_files = glob.glob(pattern)

for src_path in all_files:
    # 读取音频
    y, sr = sf.read(src_path)  # 用 soundfile 读 .ogg
    if y.ndim > 1:
        y = y[:, 0]  # 转为单通道

    # 增强
    y_enh = spectral_subtract_scipy(y, sr)

    # 保存
    rel = os.path.relpath(src_path, INPUT_ROOT)
    dst_path = os.path.join(OUTPUT_ROOT, rel).replace(".ogg", ".flac")
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, y_enh, sr)
>>>>>>> bd101112 (处理后的一部分音源)
    print(f"Processed {src_path} → {dst_path}")