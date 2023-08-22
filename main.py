import os
import time
import threading
from volumn import get_volumn
from lfasr import get_a2t_result
from record import Recorder
import record
import sys
sys.path.append('E:/StudyWork/V/ChatGLM_webui_main')
# from ChatGLM_webui_main import glminit,glminfer
from playsound import playsound
import cv2
from functools import cmp_to_key
import random

print("Import OK...")

def cmper(x,y):
    x=int(x.split(".")[0].split("\\")[-1])
    y=int(y.split(".")[0].split("\\")[-1])
    if x>y:
    	return 1
    elif y>x:
    	return -1
    else:
    	return 0


def play_video(pic_path):
    filelist = os.listdir(pic_path)
    filelist = sorted(filelist,key=cmp_to_key(cmper))
    for file in filelist:
    	file = pic_path+"/"+file
    	img = cv2.imread(file)
    	img = cv2.resize(img,(0,0),fx=1.5,fy=1.5)#
    	# img = np.zeros((1600,2560,3))
    	# img[800-h//2:800+h//2,1280-w//2:1280+w//2,:]=img_
    	cv2.imshow("img",img)
    	d = cv2.waitKey(50)
    	if d==ord("q"):
    		break

def random_play():
 	path="audio_and_video/yunyun/decoderVideo/foregroundVideo/"
 	d=dict()
 	d["01"]=[str(ind) for ind in range(14)]
 	d["02"]=[str(ind) for ind in range(17)]
 	while True:
	 	d1=random.choice(["01","02"])
	 	d2=random.choice(d[d1])
	 	pathn=f"{path}/{d1}/{d2}"
	 	play_video(pathn)
# playt=threading.Thread(target=random_play,args=())
# playt.start()

recorder=Recorder()
start_recording = False
is_recording = False
finish_recording = False
def control_record():
	global start_recording
	while True:
		# print("control_record")
		if not start_recording and not finish_recording:
			vol = get_volumn(1)
			if vol>25:
				start_recording=True
			print(vol,"start_recording:",start_recording)

t=threading.Thread(target=control_record,args=())
t.start()

def print_info():
	while True:
		print(is_recording,start_recording,finish_recording,record.finish_recording)
		time.sleep(1)
# t1=threading.Thread(target=print_info,args=())
# t1.start()

# t2=threading.Thread(target=recorder.start_record(),args=())
# t2.start()

print("Start program...")
while True:
	if not is_recording and start_recording:
		print("enter0")
		is_recording = True
		recorder.start_record()
	if is_recording:
		print("enter1")
		start_recording = record.start_recording
		finish_recording = record.finish_recording
		is_recording = record.is_recording
		print(is_recording,start_recording,finish_recording)
	if finish_recording:
		print("finish_recording")
		is_recording = False
		start_recording = False

		# audio2text
		a2t_result = get_a2t_result()
		print("a2t_result",a2t_result)

		# chat
		t=time.time()
		# chat_result = glminfer.infer(a2t_result)
		cmd=f"python E:/StudyWork/V/ChatGLM_webui_main/glminfer.py --text {a2t_result}"
		os.system(cmd)
		chat_result=" ".join(open("chat_result.txt",encoding="utf-8").readlines())
		print("chat use time:"+str(time.time()-t))
		print("chat_result",chat_result)

		# chat2audio
		t=time.time()
		python_path="E:/StudyWork/V/VITS-barbara/env/python.exe"
		vits_infer="E:/StudyWork/V/VITS-barbara/vitsinfer.py"
		cmd=f"{python_path} {vits_infer} --text {chat_result}"
		os.system(cmd)
		print("vits use time:"+str(time.time()-t))

		# playaudio
		playsound("E:/StudyWork/V/VITS-barbara/out.wav")
		time.sleep(20)

		finish_recording = False