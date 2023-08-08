# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 16:19
# @Author  : hyh
# @File    : conbine_weght.py
# @Software: PyCharm

import os

import torch

from RestNet18.net import nets
arthrosis = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
             'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
             'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
             'MIP': ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
             'Radius': ['Radius', 14],  # 桡骨
             'Ulna': ['Ulna', 12],  # 尺骨
             'PIP': ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
             'DIP': ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
             'MCP': ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）
def process():
    for weight in os.listdir('RestNet18/params'):
        # DIPFirst_best_weight.pth
        name = weight.split('.')[0]
        classes_num = arthrosis[name][1]
        net = nets.RestNet18(class_num=classes_num)
        weight_path = os.path.join('RestNet18/params', weight)
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        x = torch.randn(1, 3, 224, 224)
        store = torch.jit.trace(net, x)
        store.save(os.path.join('res18/params', f'{name}.pth'))
        print(name)


if __name__ == '__main__':
    # process()
    model = torch.jit.load(r'res18/params\DIP.pth', map_location='cpu')
    img = torch.randn((1, 3, 224, 224))
    out = model.forward(img)
    print(out.shape)
