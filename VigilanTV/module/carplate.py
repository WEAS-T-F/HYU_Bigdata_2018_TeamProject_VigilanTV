import numpy as np
import cv2
import imutils

# carplateDetecting이라는 객체를 만들 때 이미지 객체, 라벨 경로, 설정파일 경로, 가중치파일 경로를 지정
# 이미지 객체를 지정하는 곳에는 파일 이름이 아닌 opencv로 읽어들인 이미지 객체명을 써줘야 함
# ex) detect = carplateDetecting(이미지, 라벨, 설정, 가중치)
# detect 함수는 사진에 번호판을 표시해줌
# ex) image = detect.detect()
# 번호판 파싱엔 parse 함수를 사용할 것
# (주의) 파싱 함수는 아무 것도 감지하지 못할 경우 1x1의 이미지를 리턴함.

# detect = carplateDetecting(carIMG, 'cfg/obj.names' ,'cfg/yolov2-carplate.cfg', 'yolov2-carplate_2200.weights')
# saveIMG = detect.parse()


class carplateDetecting:
    def __init__(self, image, labelsPath, configPath, weightsPath):
        self.labelsPath = labelsPath
        self.configPath = configPath
        self.weightsPath = weightsPath
        self.image = image

    def detect(self, imagesize=800):
        # read labels
        # labelsPath = 'yolo-coco/obj.names'
        LABELS = open(self.labelsPath).read().strip().split("\n")

        # config와 weights 패스 등록
        # configPath = 'yolo-coco/yolov2-carplate.cfg'
        # weightsPath = 'yolo-coco/yolov2-carplate_2200.weights'

        # load darknet detector
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

        # load image with opencv
        # imagePath = 'car.jpg'
        image = self.image
        # (h, w) = image.shape[:2]
        # width_size = imagesize
        # height_size = (width_size / w) * h
        # image = cv2.resize(image, (width_size, int(height_size)), interpolation=cv2.INTER_CUBIC)
        image = imutils.resize(image, imagesize) # resize
        (H, W) = image.shape[:2] # 이미지 높이와 폭을 변수로 등록

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image

    def parse(self, imagesize=800):
        # read labels
        # labelsPath = 'yolo-coco/obj.names'
        LABELS = open(self.labelsPath).read().strip().split("\n")

        # config와 weights 패스 등록
        # configPath = 'yolo-coco/yolov2-carplate.cfg'
        # weightsPath = 'yolo-coco/yolov2-carplate_2200.weights'

        # load darknet detector
        net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

        # load image with opencv
        # imagePath = 'car.jpg'
        image = self.image
        image = imutils.resize(image, imagesize) # resize
        (H, W) = image.shape[:2] # 이미지 높이와 폭을 변수로 등록

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        plates = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                x1 = x
                y1 = y
                x2 = x1+w
                y2 = y1+h
                plate = image[y1:y2, x1:x2]
                plates.append(plate)

        if len(plates) == 0:
            return image[1:2, 1:2]
        else:
            return plates[0]