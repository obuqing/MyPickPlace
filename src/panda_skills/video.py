"""Captioned video frames for LLM -> plan -> policy demonstrations."""

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class CaptionWriter:
    def __init__(self,writer,instruction):
        self.writer=writer
        self.instruction=instruction
        self.plan='Planning before robot motion'
        self.phase='Initial scene'
        self.count=0
        self.font=_font(16)
        self.small=_font(15)

    def append_data(self,frame):
        frame=Image.fromarray(np.asarray(frame,dtype=np.uint8))
        canvas=Image.new('RGB',(512,704),(20,25,36))
        canvas.paste(frame,(0,192))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,9),'Llama 3 -> structured plan -> frozen SAC',font=self.font,fill=(240,245,255))
        y=38
        for line in textwrap.wrap(self.instruction,width=62):
            draw.text((12,y),line,font=self.small,fill=(225,230,240))
            y+=20
        draw.text((12,112),self.plan,font=self.small,fill=(120,210,255))
        draw.text((12,142),self.phase,font=self.small,fill=(180,240,160))
        draw.text((12,168),'One attempt | scripted grasp/release | learned XYZ carry',font=self.small,fill=(180,190,205))
        self.writer.append_data(np.asarray(canvas))
        self.count+=1
