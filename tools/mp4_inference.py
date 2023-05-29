import sys
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from mmdeploy_runtime import Segmentor
import numpy as np


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Player')
        self.setGeometry(100, 100, 800, 600)

        # create button
        self.select_video_button = QPushButton('Select Video', self)
        self.select_video_button.setGeometry(50, 500, 100, 50)
        self.select_video_button.clicked.connect(self.select_video)
        self.load_model_button = QPushButton('Load Model', self)
        self.load_model_button.setGeometry(200, 500, 100, 50)
        self.load_model_button.clicked.connect(self.load_model)
        self.run_button = QPushButton('Run', self)
        self.run_button.setGeometry(350, 500, 100, 50)
        self.run_button.clicked.connect(self.run_inference)
        self.play_button = QPushButton('Play', self)
        self.play_button.setGeometry(500, 500, 100, 50)
        self.play_button.clicked.connect(self.timer_play)
        self.play_button.setEnabled(False)

        # create two QLabel widgets for video display
        self.left_video_label = QLabel(self)
        self.left_video_label.setAlignment(Qt.AlignCenter)
        self.left_video_label.setFixedSize(400, 300)
        self.right_video_label = QLabel(self)
        self.right_video_label.setAlignment(Qt.AlignCenter)
        self.right_video_label.setFixedSize(400, 300)

        # create a horizontal layout to hold the video display labels
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.left_video_label)
        video_layout.addWidget(self.right_video_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_video_button)
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.play_button)

        # create a vertical layout to hold the video layout and add it to a central widget
        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(video_layout)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # create video capture objects for the two videos
        #self.cap1 = cv2.VideoCapture('test.mp4')
        #self.cap2 = cv2.VideoCapture('output_video.mp4')

        # start a timer to update the video display every 30 milliseconds
        #self.timer = QTimer(self)
        #self.timer.timeout.connect(self.update_frame)
        #self.timer.start(100)

    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.mp4 *.avi *.mov *.mkv)')
        #if self.video_path:
        #    self.play_button.setEnabled(True)

    def load_model(self):
        self.model_path = QFileDialog.getExistingDirectory(self, 'Open Model Folder', '')

    def run_inference(self):
        # create a classifier
        segmentor = Segmentor(model_path=self.model_path, device_name='cuda', device_id=0)

        # Open video capture object
        cap = cv2.VideoCapture(self.video_path)

        # Define the codec and create VideoWriter object to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('../output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            # read frame from video
            ret, frame = cap.read()

            if ret:
                # perform inference
                seg = segmentor(frame)

                # visualize inference result
                ## random a palette with size 256x3 随机生成一个256x3大小的调色板
                palette = np.random.randint(0, 256, size=(256, 3))
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color

                # convert to BGR
                color_seg = color_seg[..., ::-1]
                # frame = frame[..., ::-1]
                frame = frame * 0.5 + color_seg * 0.5
                frame = frame.astype(np.uint8)

                # write the output frame to file
                out.write(frame)

                # show the frame in a window
                #cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.play_button.setEnabled(True)

    def timer_play(self):

        # create video capture objects for the two videos
        self.cap1 = cv2.VideoCapture(self.video_path)
        self.cap2 = cv2.VideoCapture('../output_video.mp4')

        # start a timer to update the video display every 30 milliseconds
        timer = QTimer(self)
        timer.timeout.connect(self.update_frame)
        timer.start(100)

    def update_frame(self):
        # read a frame from each video capture object
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            # convert the frames to RGB format and create QImages
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            height1, width1, channel1 = frame1.shape
            bytesPerLine1 = 3 * width1
            qImg1 = QImage(frame1.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            height2, width2, channel2 = frame2.shape
            bytesPerLine2 = 3 * width2
            qImg2 = QImage(frame2.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)

            # create QPixmap objects from the QImages and set them as the labels' pixmaps
            pixmap1 = QPixmap(qImg1)
            self.left_video_label.setPixmap(pixmap1.scaled(self.left_video_label.size(), Qt.KeepAspectRatio))
            pixmap2 = QPixmap(qImg2)
            self.right_video_label.setPixmap(pixmap2.scaled(self.right_video_label.size(), Qt.KeepAspectRatio))
        else:
            # if one of the videos has ended, stop the timer and release the video capture objects
            self.timer.stop()
            self.cap1.release()
            self.cap2.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
