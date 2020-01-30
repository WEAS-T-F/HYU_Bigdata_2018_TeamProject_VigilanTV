from module.carplate import *
from module.color_identification import *
from module.extract_carplate_opencv import *
#from module.filter_addIMG import *
from module.out_result import *
import os
import numpy as np
import cv2
import math
import imutils

num = 1
endnum = 122
f = open("img/ocr_result.txt", 'w')

while num <= endnum:
    inputCarDIR = 'img/car/car(' + str(num) + ').JPG'

    print("자동차사진 : " + str(num))
    # 자동차 색 판별################################################################################################
    carIMG = cv2.imread(inputCarDIR)
    identificator = colorIdentification(carIMG)
    color = identificator.color()
    print(color)
    #print(carIMG)


    # 자동차 번호판 자르기##########################################################################################
    detect = carplateDetecting(carIMG, 'darkflow/cfg/obj.names', 'darkflow/cfg/yolov2-carplate.cfg','darkflow/cfg/yolov2-carplate_2200.weights')
    saveIMG = detect.parse()
    cv2.imwrite('img/plate/' + str(num) + '.jpg', saveIMG)


    # 번호판 이미지 전처리##########################################################################################
    inputFileName = 'img/plate/'+str(num)+'.jpg'
    outputFileName = 'img/result/'+str(num)+'.jpg'

    opencvIMG = extract_opencv(inputFileName, outputFileName)

    # 이미지 로드
    img_original = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
    # 이미지 높이, 넓이
    height, width, channel = img_original.shape
    if(height>10):
        # 이미지 확대
        img_original = cv2.resize(img_original, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        height, width, channel = img_original.shape
        # 이미지 전처리(img:canny 처리된 이미지, img2:이진화까지만 처리된 이미지)
        img, img2 = opencvIMG.img_preprocessing(img_original)
        # 직선 검출, 직선 각도 검출
        degree = opencvIMG.detect_line(img)
        # 전처리(이진화)된 이미지 회전
        cv2.imshow('img2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('height', height)
        print('width', width)
        result = opencvIMG.img_rotate(img2, degree, height, width)
        cv2.imshow('rotate', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 원본 이미지 회전
        img_original2 = opencvIMG.img_rotate(img_original, degree, height, width)
        # cv2.imshow('img_original_rotated', img_original)

        # 글자영역 찾기 - 1차
        high_x = 0
        high_y = 0
        row_x = 0
        row_y = 0
        result2, high_y, high_x, row_y, row_x = opencvIMG.find_number(result, img_original2, high_y, high_x, row_y, row_x, height,
                                                            width)
        # 글자영역 자르기 - 1차
        final_result = img_original2[row_y: high_y, row_x: high_x]
        # 1차로 자른 사진 다시 노이즈 처리
        result = opencvIMG.remove_noise(result[row_y: high_y, row_x: high_x])
        # cv2.imshow('result', result)
        # 글자영역 찾기 - 2차
        result, high_y, high_x, row_y, row_x = opencvIMG.find_number2(result, final_result, high_y, high_x, row_y, row_x, height,
                                                            width)
        # cv2.imshow('result2', result)
        # 글자영역 자르기 - 2차
        final_result = final_result[row_y: high_y, row_x: high_x]

        # 최종 이미지 수축
        kernel2 = np.ones((3, 3), np.uint8)
        result = cv2.dilate(result, kernel2, iterations=1)
        # 전처리(이진화)된 이미지 최종 결과
        # cv2.imshow('result', result)
        # 원본 이미지 최종 결과
        # cv2.imshow('final_result', final_result)

        # 원본이미지 최종결과 사이즈 조정
        final_result = cv2.resize(final_result, dsize=(432, 98), interpolation=cv2.INTER_LINEAR)

        ############### 광학문자 추출 #################
        # 이미지 로드
        # Color Change BGR to RGB
        img_original = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

        # 선명화를 위한 필터링1
        filter_img = opencvIMG.filterIMG_for_char(img_original)

        # Color Change GRAY to BGR
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR)

        # 이미지 합성을 통한 선명화
        final_fil_img = opencvIMG.addIMG2_for_char(img_original, filter_img)

        # 이미지 합성하면서 원본이미지 수정됨, 다시 불러옴
        img_original = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
        # height, width, channel = img_original.shape

        # 주영쓰 필터가 추가된 이미지 전처리
        cv2.imshow('show filtering IMG', final_fil_img)
        # 전처리
        img_2 = opencvIMG.img_preprocessing_for_char(final_fil_img)

        # Color Change GRAY to BGR
        copy_img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)

        height, width, channel = copy_img_2.shape
        # 이미지 합성으로 위한 하얀바탕 만들기
        blank_img_2 = np.zeros((height + 6, width + 6, 3), np.uint8)
        cv2.rectangle(blank_img_2, (0, 0), (width + 6, height + 6), (255, 255, 255), -1)
        # 이미지 합성
        img2_2 = opencvIMG.addIMG_for_char(blank_img_2, copy_img_2)
        # 노이즈 처리
        img3_2 = opencvIMG.remove_noise_for_char(img2_2)
        cv2.imshow("noise_filterImg", img3_2)

        result = img3_2
    else:
        result = None

    if result is not None:
        print('이미지 받아왔나? ', result)
        print(type(result))

        result = cv2.resize(result, dsize=(432, 98), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(outputFileName, result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # OCR ########################################################################################################
        ocr = out_result(outputFileName)
        resultOCR = ocr.output('1024+52font')
        txt1 = str(num) + ' : ' + str(resultOCR) + '\n'
        print(txt1)

        f.write(txt1)

        num = num + 1
        print('------------------------------------------')

    else:
        print('실패 : ', num)
        txt2 = str(num) + ' : X\n'
        f.write(txt2)
        num = num + 1
        print('------------------------------------------')
        continue

f.close()


