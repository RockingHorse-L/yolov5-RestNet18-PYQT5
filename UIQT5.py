import sys

from PyQt5.QtCore import QRect
from yolov5_master import detect
from PyQt5.QtWidgets import QTextBrowser
from infer import Detector
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # 创建打开按钮
        self.path = ''
        self.button = QPushButton('打开', self)
        self.button.setGeometry(QRect(70, 650, 75, 23))
        self.button.clicked.connect(self.open_dialog)

        self.button_2 = QPushButton('运行', self)
        self.button_2.setGeometry(QRect(500, 650, 75, 23))
        self.button_2.clicked.connect(self.test_image)

        self.button_3 = QPushButton('计算', self)
        self.button_3.setGeometry(QRect(940, 650, 75, 23))
        self.button_3.clicked.connect(self.calculate)

        # 创建显示图片的标签
        self.label = QLabel(self)
        self.label.setGeometry(QRect(90, 30, 341, 371))
        # self.label.move(150, 20)
        # self.label.resize(200, 200)

        self.label_2 = QLabel(self)
        self.label_2.setGeometry(671, 30, 351, 371)
        #self.label_2.move(150, 240)
        # self.label_2.resize(200, 20)

        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(10, 420, 1091, 192))

        # 设置窗体大小和标题
        self.setGeometry(0, 0, 1110, 776)
        self.setWindowTitle('骨龄检测')

    def open_dialog(self):
        # 打开文件选择对话框
        # file_path： 'D:/AIdata/gutou/voc/voc/VOCdevkit/VOC2007/images/1529.png'
        file_path, _ = QFileDialog.getOpenFileName(self, '打开文件', '/home')
        if file_path:
            # 加载图片到标签上
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(400, 400))
            self.path = file_path

    def test_image(self):
        image_data = detect.parse_opt(self.path)  # 假设有一个预处理函数preprocess_image，把图片转换为神经网络输入格式
        save_path = detect.main(image_data)
        #detect.main(image_data)
        pixmap = QPixmap(save_path)
        self.label_2.setPixmap(pixmap.scaled(400, 400))


    def calculate(self):
        print('开始计算骨龄')
        det = Detector()
        report = det.test()
        self.textBrowser.setText(f"结果：{report}")  # 在界面上显示测试结果

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())






