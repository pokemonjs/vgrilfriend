import pyaudio
import numpy as np

# 音频参数
chunk = 1024
sample_format = pyaudio.paInt16
channels = 2
fs = 44100


def get_volumn(seconds = 5):
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    # 录制音频
    frames = []

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # 停止和关闭音频流
    stream.stop_stream()
    stream.close()

    # 终止PyAudio库
    p.terminate()

    # 将录音数据转换成数组
    signal = np.frombuffer(b''.join(frames), dtype=np.int16)

    # 计算音频信号的RMS
    rms = np.sqrt(np.mean(np.square(signal)))

    print("音量为: ", rms)
    # return frames
    return rms


def get_volumn_from_np(frames,seconds=1):
    count = int(fs / chunk * seconds)
    if count>len(frames):
        return 35
    frames = frames[-count:]
    # 将录音数据转换成数组
    signal = np.frombuffer(b''.join(frames), dtype=np.int16)

    # 计算音频信号的RMS
    rms = np.sqrt(np.mean(np.square(signal)))

    print("实时音量为: ", rms)
    return rms

