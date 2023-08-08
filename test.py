import os

import torch

# dataset = []
# root = 'yolov5_master/runs/detect'
# img_file = os.listdir(root)
# img_path = img_file[-1]
# image_path = os.path.join(root,img_path, 'crops')
#
# file = os.listdir(image_path)
# for file_name in file:
#     path = os.path.join(image_path, file_name)
#     img_list = os.listdir(path)
#     for imgName in img_list:
#         img = imgName.split('_')
#         labels = img[0]
#         dataset.append((f'{path}//{imgName}', labels))
#         print(imgName)
#         print(img)
#         print(dataset)

SCORE = {'girl':{
    'Radius':[10,15,22,25,40,59,91,125,138,178,192,199,203, 210],
    'Ulna':[27,31,36,50,73,95,120,157,168,176,182,189],
    'MCPFirst':[5,7,10,16,23,28,34,41,47,53,66],
    'MCPThird':[3,5,6,9,14,21,32,40,47,51],
    'MCPFifth':[4,5,7,10,15,22,33,43,47,51],
    'PIPFirst':[6,7,8,11,17,26,32,38,45,53,60,67],
    'PIPThird':[3,5,7,9,15,20,25,29,35,41,46,51],
    'PIPFifth':[4,5,7,11,18,21,25,29,34,40,45,50],
    'MIPThird':[4,5,7,10,16,21,25,29,35,43,46,51],
    'MIPFifth':[3,5,7,12,19,23,27,32,35,39,43,49],
    'DIPFirst':[5,6,8,10,20,31,38,44,45,52,67],
    'DIPThird':[3,5,7,10,16,24,30,33,36,39,49],
    'DIPFifth':[5,6,7,11,18,25,29,33,35,39,49]
},
    'boy':{
    'Radius':[8,11,15,18,31,46,76,118,135,171,188,197,201,209],
    'Ulna':[25,30,35,43,61,80,116,157,168,180,187,194],
    'MCPFirst':[4,5,8,16,22,26,34,39,45,52,66],
    'MCPThird':[3,4,5,8,13,19,30,38,44,51],
    'MCPFifth':[3,4,6,9,14,19,31,41,46,50],
    'PIPFirst':[4,5,7,11,17,23,29,36,44,52,59,66],
    'PIPThird':[3,4,5,8,14,19,23,28,34,40,45,50],
    'PIPFifth':[3,4,6,10,16,19,24,28,33,40,44,50],
    'MIPThird':[3,4,5,9,14,18,23,28,35,42,45,50],
    'MIPFifth':[3,4,6,11,17,21,26,31,36,40,43,49],
    'DIPFirst':[4,5,6,9,19,28,36,43,46,51,67],
    'DIPThird':[3,4,5,9,15,23,29,33,37,40,49],
    'DIPFifth':[3,4,6,11,17,23,29,32,36,40,49]
    }
}
h = [0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5]
h = torch.tensor(h)
idx = torch.argmax(h, dim=0)
print(idx)
score = SCORE['girl']['MCPFirst'][idx]
print(score)