# yolov5-RestNet18-PYQT5
基于yolov5检测，RestNet18分类，结果用PYQT5展示的骨龄检测实验

数据集来自百度飞浆，链接如下：https://aistudio.baidu.com/aistudio/projectdetail/1485230?channelType=0&channel=0

数据集下载后，先对数据进行预处理
先运行yolov5_master/images_tag.py 并先修改里面的图片路径
![image](https://github.com/RockingHorse-L/yolov5-RestNet18-PYQT5/assets/85383732/66dd3a9b-9ad7-4f1f-b234-fe572a005892)

再打开yolov5_master/voc_to_yolo.py
修改开头的参数，分别对应类别，该项目有9个大类，作者在这个代码里做了修改不建议参考，将classes的值改为该项目的类别即可，可以参考数据集链接

![image](https://github.com/RockingHorse-L/yolov5-RestNet18-PYQT5/assets/85383732/44e94d21-5878-4a37-93d4-61fc452b5824)

同样修改里面的路径
数据集下载后，修改yolov5_master/data/mydata.yaml里的path

