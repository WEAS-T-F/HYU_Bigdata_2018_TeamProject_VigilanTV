# -*- coding: utf-8 -*-
from base_camera import BaseCamera
from module.carplate import *
from module.color_identification import *
from module.extract_carplate_opencv import *
from module.out_result import *
from main import *
import json
import datetime
import re
import requests

class Camera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""
    #imgs = [open('output.avi', 'rb').read()]

    # @staticmethod
    # def frames():
    #     cap = cv2.VideoCapture(path + saveFileName)
    #
    #     while True:
    #         ret, frame = cap.read()
    #         yield Camera.imgs[int(time.time()) % 3]

    @staticmethod
    def frames():
        print("영상URL : ", str(os.environ['MEDIA']))

        capture_path = "C:/Users/dlwlg/Desktop/capture/"
        capture_path_tmp = "C:/Users/dlwlg/Desktop/capture_tmp/"

        count = 0

        videotrack = traker()
        cap = cv2.VideoCapture(str(os.environ['MEDIA']).replace("'", ""))
        tracker = Sort()
        totalFrame = 0

        ################################
        IDcount = {}
        id_platenum = {}
        XYbox = {}
        time = {}
        print("IDcount : ")
        print(IDcount)
        print("XYbox : ")
        print(XYbox)
        ###############################
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            timetext = str(datetime.datetime.now())
            frame = cv2.putText(frame, timetext, (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, (0, 0, 255), 4)

            if ret:
                frame = np.asarray(frame)
                results = videotrack.tfnet.return_predict(frame)  # JSON
                dets = videotrack.getPoint(results)
                trackers_id = tracker.update(dets)
                tracking = videotrack.id_box(frame, trackers_id, IDcount,id_platenum)

                # ID 별로 frame 수 카운트
                num = len(trackers_id)
                for i in range(num):
                    strID = str(int(trackers_id[i][4]))
                    if strID in IDcount.keys():
                        IDcount[strID] += 1
                    else:
                        IDcount[strID] = 1

                    print("ID-" + strID + " : " + str(IDcount[strID]), end=", ")

                    # 300 프레임 (10초) 넘어가는 애들은 첫 사진 촬영 (이미지 저장)
                    if IDcount[strID] == 300:
                        time1 = datetime.datetime.now()
                        time[strID] = str(time1)
                        print(json.dumps(time, indent="\t"))
                        with open('img/time.json', 'w') as make_file:
                            json.dump(time, make_file, indent='\t')

                        XYbox[strID + "3s"] = [1, trackers_id[i][0], trackers_id[i][1], trackers_id[i][2],
                                               trackers_id[i][3]]
                        print("XYbox 90프레임 때 캡쳐 된 좌표 추가 : ")
                        print(XYbox)

                        time_parse = str(time1).split(" ")
                        parse_date = time_parse[0]
                        parse_time = time_parse[1]
                        parse_time2 = parse_time.split(".")[0]

                        parse_date = parse_date.replace("-", "")
                        parse_time2 = parse_time2.replace(":", "")

                        print("시작 날짜 :", parse_date, " 시작 시간 :", parse_time2)

                        date_start = parse_date
                        time_start = parse_time2

                        xy = XYbox[strID + "3s"]
                        topx = xy[1]
                        topy = xy[2]
                        bottomx = xy[3]
                        bottomy = xy[4]
                        image_start = date_start + time_start + "_" + strID + "_3s" + ".jpg"
                        cv2.imwrite(capture_path + image_start, tracking)
                        cropped = frame[int(topy):int(bottomy) + 1, int(topx):int(bottomx) + 1]
                        cv2.imwrite(capture_path_tmp + "box_id" + strID + "_3s" + ".jpg", cropped)

                    # 900 프레임(30초) 넘어가는 애들 증거 사진 촬영 (이미지 저장)
                    if IDcount[strID] == 900:
                        time1 = datetime.datetime.now()
                        time[strID] = str(time1)
                        print(json.dumps(time, indent="\t"))
                        with open('img/time.json', 'w') as make_file:
                            json.dump(time, make_file, indent='\t')

                        XYbox[strID + "5s"] = [2, trackers_id[i][0], trackers_id[i][1], trackers_id[i][2],
                                               trackers_id[i][3]]
                        print("XYbox 150프레임(5초) 때 캡쳐 된 좌표 추가 : ")
                        print(strID + "차량 불법주정차 시작 시각 : " + str(time1))

                        time_parse = str(time1).split(" ")
                        parse_date = time_parse[0]
                        parse_time = time_parse[1]
                        parse_time2 = parse_time.split(".")[0]

                        parse_date = parse_date.replace("-", "")
                        parse_time2 = parse_time2.replace(":", "")

                        print("탐지 날짜 :", parse_date, " 탐지 시간 :", parse_time2)

                        date_detect = parse_date
                        time_detect = parse_time2

                        print(XYbox)
                        xy = XYbox[strID + "5s"]
                        topx = xy[1]
                        topy = xy[2]
                        bottomx = xy[3]
                        bottomy = xy[4]
                        saveEvidenceCrop = "img_id" + strID + "_5s" + ".jpg"
                        saveTrackerCrop = "box_id" + strID + "_5s" + ".jpg"
                        image_detect = date_start + time_start + "_" + strID + "_5s" + ".jpg"
                        cv2.imwrite(capture_path + image_detect, tracking)
                        cropped = frame[int(topy):int(bottomy) + 1, int(topx):int(bottomx) + 1]
                        cv2.imwrite(capture_path_tmp + "box_id" + strID + "_5s" + ".jpg", cropped)
                        print(capture_path_tmp + "box_id" + strID + "_5s" + ".jpg")

                        URL = "http://192.168.1.137:8082/vigilan"
                        print("URL :", URL)
                        jsondata, data = make_json(capture_path, strID, saveTrackerCrop, image_start, image_detect,
                                                   date_start,
                                                   time_start, date_detect, time_detect)
                        print("JSON :", jsondata)
                        id_platenum[strID] = data.decode('utf-8')
                        print("번호판 : ", id_platenum[strID])
                        requests.post(URL, data=jsondata)

                totalFrame = totalFrame + 1
                tracking = cv2.resize(tracking, (1013, 549), interpolation=cv2.INTER_AREA)

                count += 1
                yield cv2.imencode('.jpg', tracking)[1].tobytes()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


def make_json(capture_path, car_id, car_capture_path, image_start, image_detect, date_start, time_start,
              date_detect,
              time_detect):
    carColor = []
    capture_path_tmp = "C:/Users/dlwlg/Desktop/capture_tmp/"
    inputCarDIR = capture_path_tmp + car_capture_path

    print("자동차사진 : " + str(car_id))
    # 자동차 색 판별################################################################################################
    print("자동차사진 경로 : ", inputCarDIR)
    carIMG = cv2.imread(inputCarDIR)
    identificator = colorIdentification(carIMG)
    color = identificator.color()
    print("차량 색상 :", color)
    carColor.append(color)
    # print(carIMG)

    # 자동차 번호판 자르기##########################################################################################
    detect = carplateDetecting(carIMG, 'cfg/obj.names', 'cfg/yolov2-carplate.cfg',
                               'cfg/yolov2-carplate_2200.weights')
    saveIMG = detect.parse()
    # cv2.imwrite('img/plate/' + str(car_id) + '.jpg', saveIMG)

    image_plate = date_start + time_start + "_" + str(car_id) + "_plate" + ".jpg"
    cv2.imwrite(str(capture_path + image_plate), saveIMG)
    print("image_plate : ", capture_path + image_plate)

    # 번호판 이미지 전처리#########################################################################################
    inputFileName = capture_path + image_plate
    outputFileName = capture_path_tmp + "plate_result_id" + str(car_id) + '.jpg'

    opencvIMG = extract_opencv(inputFileName, outputFileName)

    # 이미지 로드
    img_original = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
    # 이미지 높이, 넓이
    height, width, channel = img_original.shape

    if (height > 10):
        # 이미지 확대
        img_original = cv2.resize(img_original, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        height, width, channel = img_original.shape
        # 이미지 전처리(img:canny 처리된 이미지, img2:이진화까지만 처리된 이미지)
        img, img2 = opencvIMG.img_preprocessing(img_original)
        # 직선 검출, 직선 각도 검출
        degree = opencvIMG.detect_line(img)
        # 전처리(이진화)된 이미지 회전
        # cv2.imshow('img2', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        result = opencvIMG.img_rotate(img2, degree, height, width)
        #cv2.imshow('rotate', result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # 원본 이미지 회전
        img_original2 = opencvIMG.img_rotate(img_original, degree, height, width)
        # cv2.imshow('img_original_rotated', img_original)

        # 글자영역 찾기 - 1차
        high_x = 0
        high_y = 0
        row_x = 0
        row_y = 0
        result2, high_y, high_x, row_y, row_x = opencvIMG.find_number(result, img_original2, high_y, high_x,
                                                                      row_y,
                                                                      row_x, height,
                                                                      width)
        # 글자영역 자르기 - 1차
        final_result = img_original2[row_y: high_y, row_x: high_x]
        # 1차로 자른 사진 다시 노이즈 처리
        #cv2.imshow('rotate', result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        result = opencvIMG.remove_noise(result[row_y: high_y, row_x: high_x])
        # cv2.imshow('result', result)
        # 글자영역 찾기 - 2차
        result, high_y, high_x, row_y, row_x = opencvIMG.find_number2(result, final_result, high_y, high_x,
                                                                      row_y,
                                                                      row_x, height,
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

        ############### 광학문자 추출 ################
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
        # cv2.imshow('show filtering IMG', final_fil_img)
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
        # cv2.imshow("noise_filterImg", img3_2)

        result = img3_2
    else:
        result = None

    if result is not None:
        print('Image Load Success!!!')
        print("전체사진 :", car_capture_path)
        print("자동차 사진 :", inputCarDIR)
        print("번호판 사진 :", inputFileName)
        print("광학문자 :", outputFileName)

        result = cv2.resize(result, dsize=(1296, 294), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(outputFileName, result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # OCR #####################################################################################################
        ocr = out_result(outputFileName)
        resultOCR = ocr.output('1499lstm2+52font')
        print("OCR 1차 :", resultOCR)
        if bool(re.search('\d{2}\D\d{4}', resultOCR)):
            resultOCR = re.search('\d{2}\D\d{4}', resultOCR).group()
            print("OCR 2차 :", resultOCR)
        elif bool(re.search('\d{2}\D', resultOCR)) and bool(re.search('\d{4}', resultOCR)):
            resultOCR = re.search('\d{2}\D', resultOCR).group() + re.search('\d{4}', resultOCR).group()
            print("OCR 2차_2 :", resultOCR)

        txt1 = str(car_id) + ' : ' + str(resultOCR) + '\n'
        print(txt1)

        print('------------------------------------------')

        #########################################################################################################
        # response
        real_data = {"date_start": date_start, "time_start": time_start, "date_detect": date_detect,
                     "time_detect": time_detect, "color": color, "plate_num": resultOCR,
                     "image_start": image_start,
                     "image_detect": image_detect, "image_plate": image_plate}

        print("진짜 저장되는 데이터 : ", real_data)

        path_data = json.dumps(real_data, ensure_ascii=False)
        print(path_data)
        print("path_data :", str(path_data))
        print("plate_num : ", str(real_data['plate_num']))
        return path_data.encode('utf-8'), str(real_data['plate_num']).encode('utf-8','strict')

    else:
        print('Image Load Fail!!!')
        print("전체사진 경로:", image_start)
        print("자동차 사진 경로:", inputCarDIR)
        txt2 = str(car_id) + ' : X\n'
        print('-----------------------------------------')

        # response
        real_data = {"date_start": date_start, "time_start": time_start, "date_detect": date_detect,
                     "time_detect": time_detect, "color": color, "plate_num": "X", "image_start": image_start,
                     "image_detect": image_detect, "image_plate": image_plate}

        path_data = json.dumps(real_data, ensure_ascii=False)
        print(path_data)
        print("path_data :", str(path_data))
        print("plate_num : ", str(real_data['plate_num']))
        return path_data.encode('utf-8'), str(real_data['plate_num']).encode('utf-8','strict')

