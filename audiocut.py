import librosa
import numpy as np
import soundfile as sf

def split_audio_into_chunks(audio_file_path, chunk_length=5.0, output_folder='chunks'):
    # 加载音频文件
    y, sr = librosa.load(audio_file_path, sr=None)
    
    # 计算每个片段的样本数
    chunk_samples = int(chunk_length * sr)
    
    # 创建输出文件夹
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 计算总片段数
    total_samples = len(y)
    num_chunks = int(np.ceil(total_samples / chunk_samples))
    
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = start_sample + chunk_samples
        
        # 如果最后一个片段不足5秒，则使用上一个片段的部分进行补全
        if end_sample > total_samples:
            end_sample = total_samples
            start_sample = end_sample - chunk_samples
        
        chunk = y[start_sample:end_sample]
        
        # 保存片段
        output_file_path = os.path.join(output_folder, f'chunk_{i+1}.ogg')
        sf.write(output_file_path, chunk, sr)

# 测试运行正常与否
audio_file_path = './train_audio/21038/iNat65519.ogg'
split_audio_into_chunks(audio_file_path)
# 测试通过