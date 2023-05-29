import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from mmdeploy_runtime import Segmentor
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Inference')
        self.setGeometry(100, 100, 800, 600)
        self.image_path = ''
        self.model_path = ['']
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 300, 300)
        self.result_label = QLabel(self)
        self.result_label.setGeometry(450, 50, 300, 300)
        self.image_label.setText('No image loaded')
        self.result_label.setText('No result yet')
        self.load_image_button = QPushButton('Load Image', self)
        self.load_image_button.setGeometry(50, 500, 200, 50)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_model_button = QPushButton('Load Model', self)
        self.load_model_button.setGeometry(300, 500, 200, 50)
        self.load_model_button.clicked.connect(self.load_model)
        self.run_button = QPushButton('Run', self)
        self.run_button.setGeometry(550, 500, 200, 50)
        self.run_button.clicked.connect(self.run_inference)

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg)')
        pixmap = QPixmap(self.image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def load_model(self):
        self.model_path = QFileDialog.getExistingDirectory(self, 'Open Model Folder', '')

    def run_inference(self):
        img = cv2.imread(self.image_path)

        # create a classifier
        segmentor = Segmentor(model_path=self.model_path, device_name='cuda', device_id=0)
        # perform inference
        seg = segmentor(img)

        # visualize inference result
        ## random a palette with size 256x3 随机生成一个256x3大小的调色板
        palette = np.random.randint(0, 256, size=(256, 3))
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]
        # img = img[..., ::-1]
        img = img * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        cv2.imwrite('../output_segmentation.png', img)
        pixmap = QPixmap('../output_segmentation.png')
        self.result_label.setPixmap(pixmap)
        self.result_label.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())