import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_file_path, output_image_path):
    # 加载音频文件
    y, sr = librosa.load(audio_file_path, sr=None)
    
    # 生成梅尔频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 绘制声谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    # 保存声谱图为图像文件
    plt.savefig(output_image_path)
    plt.close()

# 测试运行正常与否
audio_file_path = './train_audio/21038/iNat65519.ogg'
output_image_path = 'output_spectrogram.png'
create_spectrogram(audio_file_path, output_image_path)
# 测试通过