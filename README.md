# 整体架构

实时录音+语音识别+大模型对话+语音合成

## 实时录音

PyAudio实现，结合两秒内的音量判断开始录音和结束

相关程序：record.py volumn.py

## 语音识别

打算自己部署模型，但是没找到合适的，发现API服务这里很成熟，直接调用API

评测了两款，百度智能云和讯飞

一言难尽，智能云我传一段普通话给我识别出啦一串哼哼哈哈，我不知道是不是打开方式有问题，讯飞返回的json很离谱，对象里面套字符串，这个字符串解析出来对象里面又套字符串，解析json那段代码至少写了4，5个json.loads

相关程序：audio2text.py lfasr.py

## 大模型对话

本地部署了清华的ChatGLM，6G显存吃的一口不剩，3060推理时间4分钟

相关程序：glminit.py glminfer.py

## 语音合成

用VITS模型微调，音频是自己从动漫里扒拉的，一小时的特别篇，就凑出来3分钟的训练素材，我差点原神启动扒拉九条沙罗语音包（同款cv），但是想起来枫丹更新包40G，对不起，打扰了

相关程序：vitsinfer.py