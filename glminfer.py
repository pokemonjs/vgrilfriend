import os
import time
import sysconfig
import sys
# from modules import options

from modules.model import load_model

from modules.options import cmd_opts
from modules.ui import create_ui
import argparse

# patch PATH for cpm_kernels libcudart lookup
os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + os.path.join(sysconfig.get_paths()["purelib"], "torch\lib")


def predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
    ctx.limit_round()
    flag = True
    from modules.model import infer
    for _, output in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            use_stream_chat=use_stream_chat
    ):
        if flag:
            ctx.append(query, output)
            flag = False
        else:
            ctx.update_last(query, output)
        yield ctx.rh, ""
    ctx.refresh_last()
    yield ctx.rh, ""

def infer(input_message = "你好啊，今天天气真好"):
    # from modules.ui import predict
    from modules.context import Context
    ctx = Context()
    max_length = 2048
    top_p = 0.7
    temperature = 0.95
    use_stream_chat = False
    pg = predict(ctx,input_message,max_length,top_p,temperature,use_stream_chat)
    ret = next(pg)
    print("ret:",ret)
    return ret[0][0][1]
    # print(pg,type(pg))
    # print(next(pg))

import glminit

res = infer(cmd_opts.text)
print("res:",res)
with open("chat_result.txt","w",encoding='utf-8') as f:
    f.write(res)