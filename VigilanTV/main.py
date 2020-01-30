from darkflow.net.build import TFNet
import cv2
import numpy as np
from sort.sort import *
import time
from PIL import Image
import os

class traker:

    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.5, "gpu":0.3}  # threshold가 신뢰율임
    tfnet = TFNet(options)  # JSON

    # 사실상 얘는 안쓰는 함수임
    def boxing(self, original_img, predictions):
        newImage = np.copy(original_img)

        for result in predictions:
            if result['label'] != 'car':
                continue
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence > 0.25:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                       (0, 230, 0), 1, cv2.LINE_AA)

        return newImage


    # 얘 쓰는 함수임
    def getPoint(self, predictions):
        bboxes = []
        for result in predictions:
            if result['label'] != 'car':
                continue
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            score = result['confidence']  # 신뢰율

            bbox = [top_x, top_y, btm_x, btm_y, score]  # 5개까지만 sort가 읽음
            bboxes.append(bbox)

        bboxes = np.array(bboxes)  # sort 적용하려면 numpy array로 넣어줘야함 (5개)

        return bboxes

    def id_box(self, image, boxes):
        image = np.copy(image)

        for box in boxes:
            top_x = int(box[0])
            top_y = int(box[1])
            btm_x = int(box[2])
            btm_y = int(box[3])

            track_id = str(int(box[4]))

            image = cv2.rectangle(image, (top_x, top_y), (btm_x, btm_y), (255, 255, 255), 3)
            image = cv2.putText(image, track_id, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255),1)

        return image
