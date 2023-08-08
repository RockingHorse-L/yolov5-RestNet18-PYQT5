# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.bone_pred import predChosse


figer_labels =['DIPFirst',
            'DIPThird',
            'DIPFifth',
    
            'MIPThird',
            'MIPFifth',

            'PIPFirst',
            'PIPThird',
            'PIPFifth',

            'MCPThird',
            'MCPFifth',

            'MCPFirst',

            'Ulna',
            'Radius'
            ]

@smart_inference_mode()
# 'D:/AIdata/person/images'
def run(
        weights=ROOT / 'runs/train/exp/weights/best.pt',  # model path or triton URL
        # 'D:/AIdata/gutou/voc/voc/VOCdevkit/VOC2007/images'
        source=ROOT / 'data/imgaes/1589.png', # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/mydata.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=True,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='D:/AIdata/gutou/voc/voc/VOCdevkit/VOC2007/testEndImg/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride

):

    """
        这段代码主要是处理输入的来源，定义了一些布尔值区分输入的是图片、视频、网络流还是摄像头
    """

    # 输入的路径变为字符串
    source = str(source)
    # 是否保存图片和txt文件，如果nosave传入为false且source结尾不是txt则保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断source是不是视频/图像文件路径
    # path()提取文件名。suffix: 最后一个组件的文件扩展名。若source是“D:/DATA/1.jpg 则Path(source).suffix是”.jpg“
    # Path(source).suffix[1:]是jpg
    # IMG_FORMATS + VID_FORMATS两个变量保存的是所有视频和图片的格式后最
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否是连接
    # .lower转为小写，.upper()转为大写 .title()首字符转为大写其余为小写, .startwith('http://')返回Ture or Flase
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断source是否有摄像头
    # .isnumeric()是否由数字组成 ，返回Ture Flase
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # 返回文件，如果source是指向一个图片/视频的连接，则下载数据
        source = check_file(source)  # download

    """
        主要是保存结果的路径
    """
    # Directories
    # 保存运行结构的文件夹 save_dir：WindowsPath('runs/detect/exp4')
    # incerement_path()判断目录是否存在，如果存在则在名称后面添加数字
    # project：WindowsPath('runs/detect')
    # name：'exp' exist_ok：
    # Path(project) / name目录拼接
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # exist_ok 如果为False表示目录已经存在 exist_ok为True则不会抛出异常而直接使用存在的目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    """
        加载模型
    """
    # Load model device：device(type='cuda', index=0)
    device = select_device(device)
    # DetectMultiBackend定义在models.common模块中是我么们要加载的网络 weights是权重文件
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
    # names：就是我们分类的标签名，比如摔倒检测的值是{0: 'fall', 1: 'person', 2: '10+'}
    # pt: 加载的是否是pytorch模型（也就是pt格式的文件）
    # jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
    # onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
    # model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。


    # 确保图片的尺寸能整除stride = 32(默认32) 如果不能则调为整数再返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    """
        判断是否是视频流或者是图片流
        如果是视频流则使用LoadStreams加载视频流
        否则使用LoadImage
    """
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # dt存储每一步的耗时
    # seen 计数功能，已经处理完了多少帧图片
    """
        对模型进行预处理
        遍历dataset传回来的值，整理图片信息
        进行预测，根据run函数里面的置信度和IOU进行信息过滤
        转为tensor格式、归一化、并为批处理增加一维
    """
    for path, im, im0s, vid_cap, s in dataset:
        # 有一些任务，可能事先需要设置，事后做清理工作。对于这种场景，Python的with语句提供了一种非常方便的处理方式。
        # 一个很好的例子是文件处理，你需要获取一个文件句柄，从文件中读取数据，然后关闭文件句柄。
        with dt[0]:
            # 由numpy转为tensor
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        # 得到网络模型的预测框pred 是二维list
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        # 对网络模型的预测框进行筛选
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            det = predChosse(det)
            box = det
            seen += 1

            # 如果输入源是webcam 遍历每个图像并复制一份备份到变量im0中，同时将当前图像的路径和计数器记录到p, frame中
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            # 不是的话则直接使用im0s变量中的图片，同时将当前图像的路径和计数器记录到p, frame中
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 将图像路径转为path对象
            p = Path(p)  # to Path
            # 设置保存路径 'runs\\detect\\exp5\\bus.jpg'
            save_path = str(save_dir / p.name)  # im.jpg
            # 设置保存框坐标的txt文件路径，每张图对应一个框坐标信息 'runs\\detect\\exp5\\labels\\bus'
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 设置输出图片信息。图片shape(w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            # 得到原图的w, h
            # im0.shape
            # (1080, 810, 3)
            # gn: tensor([ 810, 1080,  810, 1080])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 如果save_crop为true则将检测到的boundin_box单独保存为一张图片
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 得到一个绘图的类，类中存储了原图， 线条宽度， 类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将标注了bounding_box大小调整为和原图一致
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 获取类别数量 unique()去重函数
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 打印目标检测结果的操作
                i = 0
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框+将预测到的目标剪切出来保存成图片
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{figer_labels[i]}_{p.stem}.jpg', BGR=True)
                        i += 1
                    if i > 12:
                        i = 0

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return save_path

def parse_opt(img):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=img, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=True, help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    save_path = run(**vars(opt))
    return save_path

if __name__ == '__main__':
    opt = parse_opt(r'D:\AIdata\gutou\voc\voc\VOCdevkit\VOC2007\images')
    save_path = main(opt)
