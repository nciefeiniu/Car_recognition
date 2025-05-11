#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/10 22:56
# @File    : gui.py
import shutil
import traceback
import warnings
import torch
import threading
import os.path as osp
from PyQt5.QtGui import *
from pathlib import Path
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import os
# from ultralytics import YOLOv10
import sys
from PyQt5.QtCore import QTimer, Qt
import cv2
# import detect_tools as tools
# from paddleocr import PaddleOCR

from ui import Ui_MainWindow


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stop = False
        self.file_path = ""
        # 图片读取进程
        self.output_size = 640
        self.img2predict = ""
        # 更新视频图像
        self.timer_camera = QTimer()
        self.cap = None
        self.is_camera_open = False
        self.stopEvent = threading.Event()
        # 加载检测模型
        self.model = None
        self.conf_threshold = 0.25

        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.pushButton.clicked.connect(self.upload_img)
        self.pushButton_2.clicked.connect(self.detect_img)

        self.pushButton_5.clicked.connect(self.vedio_show)
        # self.pushButton_9.clicked.connect(self.camera_show)
        self.pushButton_10.clicked.connect(self.video_stop)
        self.pushButton_3.clicked.connect(self.close_window)

        self.reset_vid()

    def upload_img(self):
        """上传图片"""
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            self.file_path = fileName
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一放在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.label_5.setScaledContents(True)
            self.label_5.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.label_6.setText(' ')

    def detect_img(self):
        """检测图片"""
        org_path = self.file_path
        # 目标检测
        try:
            from detect import detect_image
            now_img = detect_image(org_path)
        except:
            print(traceback.format_exc())
        # results = self.model.predict(source=org_path, imgsz=self.output_size, conf=self.conf_threshold)
        # now_img = results[0].plot(labels=False)
        # location_list = results[0].boxes.xyxy.tolist()
        # location_list = [list(map(int, e)) for e in location_list]
        # plates = []
        # for i in range(len(location_list)):
        #     x1, y1, x2, y2 = location_list[i]
        #     im1 = now_img[y1 - 10:y2 + 10, x1 - 10:x2 + 10]
        #     ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        #     result = ocr.ocr(im1, cls=True)
        #     if len(result) > 0 and result[0] is not None:
        #         plate = result[0][0][1][0]
        #         plate = plate.replace('·', ' ')
        #         plates.append(plate)
        # for index, location in enumerate(location_list):
        #     h = (location[3] - location[1]) // 2 + 15
        #     position = [location[0], location[1] - h, location[2], location[3] - h]
        #     now_img = tools.cv2AddChineseText(now_img, plates[index], position)
        cv2.imwrite("images/tmp/single_result.jpg", now_img)
        self.label_6.setScaledContents(True)
        self.label_6.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        self.reset_vid()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            # 目标检测
            results = self.model.predict(source=now_img, imgsz=self.output_size, conf=self.conf_threshold)
            now_img = results[0].plot(labels=False)
            location_list = results[0].boxes.xyxy.tolist()
            location_list = [list(map(int, e)) for e in location_list]
            plates = []
            for i in range(len(location_list)):
                x1, y1, x2, y2 = location_list[i]
                im1 = now_img[y1 - 10:y2 + 10, x1 - 10:x2 + 10]
                ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                result = ocr.ocr(im1, cls=True)
                if len(result) > 0 and result[0] is not None:
                    plate = result[0][0][1][0]
                    plate = plate.replace('·', ' ')
                    plates.append(plate)
            for index, location in enumerate(location_list):
                h = (location[3] - location[1]) // 2 + 15
                position = [location[0], location[1] - h, location[2], location[3] - h]
                now_img = tools.cv2AddChineseText(now_img, plates[index], position)
            cv2.imwrite("images/tmp/single_result_vid.jpg", now_img)
            self.label_8.clear()
            self.label_8.setScaledContents(True)
            self.label_8.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            print('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            print('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
        else:
            print('摄像头未开启')
            self.label_8.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.label_8.clear()

    def close_window(self):
        self.stopEvent.set()
        self.close()

    def reset_vid(self):
        """界面重置事件"""
        self.pushButton_5.setEnabled(True)
        # self.pushButton_9.setEnabled(True)
        # self.label_8.setText('检测窗口')


if __name__ == '__main__':
    # todo 修改模型权重路径
    model_dir = 'runs/detect/train/weights/best.pt'

    #  解决2K等高分辨率屏幕不匹配或自适应问题，导致部分控件显示不完全
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    my_window = MyWindow(model_dir)
    my_window.show()
    sys.exit(app.exec_())