# -*- coding: utf-8 -*-
import os
import shutil

src_dir = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\projects\图片"
dst_dir = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\projects\阶段性汇报7.15\projects\LumiMind-Project-Summary_ppt169_20260715\svg_output\images"

os.makedirs(dst_dir, exist_ok=True)

name_map = {
    "光显示界面.png": "light_display.png",
    "AI问答.png": "ai_chat.png",
    "心率.png": "heart_rate.png",
    "成员界面.png": "member_view.png",
    "知识库.png": "knowledge_base.png",
    "ecg_waveform.png": "ecg_waveform.png",
    "spectrum.png": "spectrum.png",
}

for orig, eng in name_map.items():
    s = os.path.join(src_dir, orig)
    d = os.path.join(dst_dir, eng)
    shutil.copy2(s, d)
    print(f"OK {eng} ({os.path.getsize(d)} bytes)")

print("done")
