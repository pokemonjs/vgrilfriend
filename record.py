import pyaudio

import wave
from volumn import get_volumn_from_np

# 设置录音参数

CHUNK = 1024

FORMAT = pyaudio.paInt16

CHANNELS = 1

RATE = 16000

RECORD_SECONDS = 5

WAVE_OUTPUT_FILENAME = "record.wav"
# WAVE_OUTPUT_FILENAME = "record.pcm"
is_recording=False
start_recording=False
finish_recording=False

class Recorder():
	def start_record(self):
		self.start = True
		start_recording=True
		is_recording=True
		# 初始化PyAudio

		self.p = pyaudio.PyAudio()

		# 打开音频流

		self.stream = self.p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True)

		print("* 录音中...")

		# 录音数据缓存

		self.frames = []

		# 录音

		# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		while True:
			data = self.stream.read(CHUNK)
			self.frames.append(data)
			vol = get_volumn_from_np(self.frames)
			if vol < 15:
				self.start=False
			# print(self.start)
			if self.start==False:
				self.end_record()
				# time.sleep(2)
				break

	def end_record(self):

		# 停止数据流
		self.stream.stop_stream()
		self.stream.close()
		# 关闭PyAudio
		self.p.terminate()

		# 写入录音文件
		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(self.p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(self.frames))
		wf.close()
		
		global is_recording,start_recording,finish_recording
		is_recording=False
		start_recording=False
		finish_recording=True
		print("* finish_recording:",finish_recording)
		print("* 录音结束")