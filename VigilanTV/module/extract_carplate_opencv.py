# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import imutils


#img_number = '12'
#inputFileName = 'realdata2/'+img_number+'.jpg'
#outputFileName = 'realoutput2/'+img_number+'.jpg'
#noise = 0

class extract_opencv:
    def __init__(self, inputDIR, outputDIR):
        self.inputDIR = inputDIR
        self.outputDIR = inputDIR

    def remove_noise(self, img_edge2):
        img = img_edge2

        cnt, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_height, img_width = img.shape
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)

            if (h < (img_height * 0.1)) or (w < (img_width * 0.05)):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

        return img

    def img_preprocessing(self, img):
        # 이미지 로드
        img_original = img

        # 이미지 흑백화
        imgray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        imgray = cv2.bilateralFilter(imgray, 10, 100, 100)
        # cv2.imshow('blur', imgray)

        # 이미지 수축
        kernel2 = np.ones((3, 3), np.uint8)
        imgray = cv2.dilate(imgray, kernel2, iterations=1)

        imgray = cv2.bilateralFilter(imgray, 10, 120, 120)
        # cv2.imshow('equalizeHist', imgray)

        # 이미지 이진화
        imgray = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)

        # 이미지 팽창
        kernel = np.ones((5, 5), np.uint8)
        imgray = cv2.erode(imgray, kernel, iterations=1)

        # 이미지 윤곽선 따기 - 직선 검출을 위한 전처리
        img_edge = cv2.Canny(imgray, 50, 100, 3)

        # cv2.imshow('img_preprocessing', img_edge)

        return img_edge, imgray

    def detect_line(self, img):
        lines = cv2.HoughLines(img, 1, np.pi / 180, 50)

        if lines is None:
            print("Do not Detect Lines")
            return -90;
        else:
            for rho, theta in lines[0]:
                x1 = int(np.cos(theta) * rho + 1000 * (-np.sin(theta)))
                y1 = int(np.sin(theta) * rho + 1000 * np.cos(theta))
                x2 = int(np.cos(theta) * rho - 1000 * (-np.sin(theta)))
                y2 = int(np.sin(theta) * rho - 1000 * np.cos(theta))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                rad = math.atan2(x1 - x2, y1 - y2)
                angle = (rad * 180) / np.pi
                print(angle)

            return angle

    def img_rotate(self, img, degree, height, width):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -(degree + 90), 1)
        dst = cv2.warpAffine(img, matrix, (width, height))

        return dst

    def find_number(self, img_edge2, img_original, high_y, high_x, row_y, row_x, height, width):
        high_x = 0
        high_y = 0
        row_x = 0
        row_y = 0

        cnt, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        box_point = []

        first = 0
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio2 = float(h) / w

            # if (aspect_ratio>=0.2)and(aspect_ratio<=2.0)and(rect_area>=100)and(rect_area<=700):
            if (aspect_ratio2 >= 0.3) and (h > (height * 0.3)) and (w < (width * 0.2)) and (w > (width * 0.04)):
                # cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if (first == 0):
                    row_x = x
                    row_y = y
                    first = 1

                if (x + w > high_x):
                    high_x = x + w
                if (x < row_x):
                    row_x = x
                if (y + h > high_y):
                    high_y = y + h
                if (y < row_y):
                    row_y = y

                continue;
            else:
                # cv2.rectangle(img_original, (x, y), (x + w, y + h), (100, 100, 100), 2)
                continue;

        # cv2.rectangle(img_original, (row_x, row_y), (high_x, high_y), (0, 255, 0), 1)
        box_point.append(cv2.boundingRect(cnt))

        return img_original, high_y, high_x, row_y, row_x

    def find_number2(self, img_edge2, img_original, high_y, high_x, row_y, row_x, height, width):
        high_x = 0
        high_y = 0
        row_x = 0
        row_y = 0

        cnt, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        first = 0
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio2 = float(h) / w
            rect_area = w*h

            if (aspect_ratio2 >= 1) and (aspect_ratio2 <= 3.5) and (rect_area >= 100):
                cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if (first == 0):
                    row_x = x
                    row_y = y
                    first = 1

                if (x + w > high_x):
                    high_x = x + w
                if (x < row_x):
                    row_x = x
                if (y + h > high_y):
                    high_y = y + h
                if (y < row_y):
                    row_y = y
                continue;
            else:
                # cv2.rectangle(img_original, (x, y), (x + w, y + h), (100, 100, 100), 2)
                continue;

        return img_original, high_y, high_x, row_y, row_x

    def remove_noise_for_char(self, img_edge2):
        img = img_edge2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 이미지 팽창
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        cnt, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_height, img_width = img.shape

        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)

            if (h < (img_height * 0.1)) or (w < (img_width * 0.05)):
                if h < img_height * 0.7:
                    idx = i  # The index of the contour that surrounds your object
                    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
                    cv2.drawContours(mask, contours, idx, 255, -1)  # Draw filled contour in mask
                    out = np.zeros_like(img)  # Extract out the object and place into output image
                    out[mask == 255] = img[mask == 255]

                    bnw = img[mask == 255]
                    black = 0
                    white = 0

                    for j in range(len(bnw)):
                        if bnw[j] == 0:
                            black += 1
                        else:
                            white += 1

                    if (black * 2) > white:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

            else:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
                continue

        return img

    def img_preprocessing_for_char(self, img):
        # 이미지 로드
        img_original = img

        # 이미지 흑백화
        imgray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        # 이미지 수축
        kernel2 = np.ones((5, 5), np.uint8)
        imgray = cv2.dilate(imgray, kernel2, iterations=1)

        imgray = cv2.bilateralFilter(imgray, 10, 100, 100)

        # 이미지 이진화
        imgray = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

        # 이미지 팽창
        kernel = np.ones((2, 2), np.uint8)
        imgray = cv2.erode(imgray, kernel, iterations=1)

        return imgray

    def addIMG_for_char(self, imgdir1, imgdir2):
        # 이미지 합성
        img1 = imgdir1  # 원본 애
        img2 = imgdir2  # 필터 씌운애

        #cv2.imshow('addIMG_for_char1', img1)
        #cv2.imshow('addIMG_for_char2', img2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # I want to put logo on top-left corner, So I create a ROI
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 255, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img1[3:rows + 3, 3:cols + 3] = dst

        return img1

    def filterIMG_for_char(self, imgdir):
        # 이미지 필터
        image = imgdir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img2 = clahe.apply(gray)

        return img2

    def addIMG2_for_char(self, imgdir1, imgdir2):
        # 이미지 합성
        img1 = imgdir1  # 원본 애
        img2 = imgdir2  # 필터 씌운애

        #cv2.imshow('img1', img1)
        #cv2.imshow('img2', img2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # I want to put logo on top-left corner, So I create a ROI
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 170, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst

        return img1


#
# # 이미지 로드
#
# img_original = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
#
# hist_full = cv2.calcHist([img_original], [1], None, [256], [0, 256])
# print("histogram", hist_full)
# # 이미지 확대
# img_original = cv2.resize(img_original, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
# height, width, channel = img_original.shape
#
# print('height', height)
# print('width', width)
#
#
# # 이미지 전처리
# img = img_preprocessing(img_original)
# # 직선 검출, 직선 각도 검출
# degree = detect_line(img)
# result = img_rotate(noise, degree)
#
# cv2.imshow('rotate', result)
#
# high_x = 0
# high_y = 0
# row_x = 0
# row_y = 0
#
# ####################################################
#
#
# # 이미지 수축
# kernel2 = np.ones((3, 3), np.uint8)
# result = cv2.dilate(result, kernel2, iterations=1)
# cv2.imshow('result', result)
#
#
# img_original2 = img_rotate(img_original, degree)
# result = removeNoise(result)
# cv2.imshow('noise', result)
#
# result2 = find_number(result, img_original2)
# cv2.imshow('img_original', img_original)
# cv2.imshow('find_number', result2)
#
# result = removeNoise(result[row_y: high_y, row_x: high_x])
# cv2.imshow('-----', result)
#
# #result = contour3(result)
# #cv2.imshow('img_original_noise2', result)
#
# result = cv2.resize(result,  dsize=(432, 98), interpolation=cv2.INTER_LINEAR)
#
# cv2.imwrite(outputFileName, result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()