from module.carplate import *
from module.color_identification import *
from module.extract_carplate_opencv import *
from module.out_result import *
from flask import Flask, request, make_response
from main import *
import json
import cv2
from sort.sort import *
import datetime
import re

app = Flask(__name__)

saveTrackerCrop = ""
@app.route('/getPath', methods = ['POST', 'GET'])
def getPath():
    print("Success connected")
    if request.method == 'POST':
        capture_path = "C:/Users/dlwlg/Desktop/capture/"

        json_path = request.get_json()
        path = json_path['serverPath']
        saveFileName = json_path['saveFileName']
        print(json_path['serverPath'])
        print(json_path['saveFileName'])
        ###################################################################################################################################
        videotrack = traker()
        cap = cv2.VideoCapture(path + saveFileName)
        tracker = Sort()
        totalFrame = 0
        # 재생할 파일의 넓이 얻기
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # 재생할 파일의 높이 얻기
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # 재생할 파일의 프레임 레이트 얻기
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 저장할 비디오 코덱
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # 저장할 파일 이름
        output_avi = 'output.avi'

        # 파일 stream 생성
        out = cv2.VideoWriter(capture_path + output_avi, fourcc, fps, (int(width), int(height)))
        # filename : 파일 이름
        # fourcc : 코덱
        # fps : 초당 프레임 수
        # width : 넓이
        # height : 높이

        #################################
        IDcount = {}
        XYbox = {}
        time = {}
        print("IDcount : ")
        print(IDcount)
        print("XYbox : ")
        print(XYbox)
        ################################
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
                (H, W) = frame.shape[:2]

                frame = np.asarray(frame)
                results = videotrack.tfnet.return_predict(frame)  # JSON
                dets = videotrack.getPoint(results)
                trackers_id = tracker.update(dets)
                tracking = videotrack.id_box(frame, trackers_id)

                # ID 별로 frame 수 카운트
                num = len(trackers_id)
                for i in range(num):
                    print(trackers_id[i][4])
                    strID = str(trackers_id[i][4])
                    if strID in IDcount.keys():
                        IDcount[strID] += 1
                    else:
                        IDcount[strID] = 1

                    # 90 프레임 (3초) 넘어가는 애들은 첫 사진 촬영 (이미지 저장)
                    if IDcount[strID] == 150:
                        XYbox[strID + "3s"] = [1, trackers_id[i][0], trackers_id[i][1], trackers_id[i][2], trackers_id[i][3]]
                        print("XYbox 90프레임 때 캡쳐 된 좌표 추가 : ")
                        print(XYbox)
                        xy = XYbox[strID + "3s"]
                        topx = xy[1]
                        topy = xy[2]
                        bottomx = xy[3]
                        bottomy = xy[4]
                        cv2.imwrite(capture_path + "img_id" + strID + "_3s" + ".jpg", tracking)
                        cropped = frame[int(topy):int(bottomy) + 1, int(topx):int(bottomx) + 1]
                        cv2.imwrite(capture_path + "img_id" + strID + "_3s" + ".jpg", cropped)

                    # 150 프레임(5초) 넘어가는 애들 증거 사진 촬영 (이미지 저장)
                    if IDcount[strID] == 150:
                        time1 = datetime.datetime.now()
                        time[strID] = str(time1)
                        print(json.dumps(time, indent="\t"))
                        with open('img/time.json', 'w') as make_file:
                            json.dump(time, make_file, indent='\t')

                        XYbox[strID + "5s"] = [2, trackers_id[i][0], trackers_id[i][1], trackers_id[i][2], trackers_id[i][3]]
                        print("XYbox 150프레임(5초) 때 캡쳐 된 좌표 추가 : ")
                        print(strID + "차량 불법주정차 시작 시각 : " + str(time1))
                        print(XYbox)
                        xy = XYbox[strID + "5s"]
                        topx = xy[1]
                        topy = xy[2]
                        bottomx = xy[3]
                        bottomy = xy[4]
                        saveEvidenceCrop = "img_id" + strID + "_5s" + ".jpg"
                        saveTrackerCrop = "box_id" + strID + "_5s" + ".jpg"
                        cv2.imwrite(capture_path + "img_id" + strID + "_5s" + ".jpg", tracking)
                        cropped = frame[int(topy):int(bottomy) + 1, int(topx):int(bottomx) + 1]
                        cv2.imwrite(capture_path + "box_id" + strID + "_5s" + ".jpg", cropped)
                        print(capture_path + "box_id" + strID + "_5s" + ".jpg")

                totalFrame = totalFrame + 1

                # Display the resulting frame
                cv2.imshow('frame', tracking)
                # print(totalFrame)
                out.write(tracking)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        total = 0
        idlist = []
        # 불법 주정차 차량 판단하기#####################################
        for id, fps in IDcount.items():
            if fps >= 150:  # 150 (5초 이상 일때)
                print("ID : " + id + "는 불법 주정차 차량입니다.")
                print("사각형 좌표 : ")
                total = total + 1
                idlist.append(id)
        ###########################################################

        print("IDcount : ")
        print(IDcount)

        print("XYbox : ")
        print(XYbox)

        print("idlist 불법주정차 : ", idlist)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


        ###################################################################################################################################
        num = 0
        endnum = total - 1
        print("불법주정차 차량 숫자 : ", total)
        #idlist[num]
        carColor = []
        f = open("img/ocr_result.txt", 'w')

        while num <= endnum:

            inputCarDIR = capture_path + "box_id" + str(idlist[num]) + '_5s.jpg'

            print("자동차사진 : " + str(idlist[num]))
            # 자동차 색 판별################################################################################################
            print("자동차사진 경로 : ", inputCarDIR)
            carIMG = cv2.imread(inputCarDIR)
            identificator = colorIdentification(carIMG)
            color = identificator.color()
            print("차량 색상 :", color)
            carColor.append(color)
            #print(carIMG)


            # 자동차 번호판 자르기##########################################################################################
            detect = carplateDetecting(carIMG, 'cfg/obj.names', 'cfg/yolov2-carplate.cfg','cfg/yolov2-carplate_2200.weights')
            saveIMG = detect.parse()
            #cv2.imwrite('img/plate/' + str(idlist[num]) + '.jpg', saveIMG)
            cv2.imwrite(capture_path + "plate_id" + str(idlist[num]) + '.jpg', saveIMG)

            # 번호판 이미지 전처리##########################################################################################
            inputFileName = capture_path + "plate_id" + str(idlist[num]) + '.jpg'
            outputFileName = capture_path + "plate_result_id" + str(idlist[num]) + '.jpg'

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
                #cv2.imshow('img2', img2)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
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
                result2, high_y, high_x, row_y, row_x = opencvIMG.find_number(result, img_original2, high_y, high_x, row_y,
                                                                              row_x, height,
                                                                              width)
                # 글자영역 자르기 - 1차
                final_result = img_original2[row_y: high_y, row_x: high_x]
                # 1차로 자른 사진 다시 노이즈 처리
                result = opencvIMG.remove_noise(result[row_y: high_y, row_x: high_x])
                # cv2.imshow('result', result)
                # 글자영역 찾기 - 2차
                result, high_y, high_x, row_y, row_x = opencvIMG.find_number2(result, final_result, high_y, high_x, row_y,
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
                #cv2.imshow('show filtering IMG', final_fil_img)
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
                #cv2.imshow("noise_filterImg", img3_2)

                result = img3_2
            else:
                result = None

            if result is not None:
                print('Image Load Success!!!')
                print("전체사진 :", saveEvidenceCrop)
                print("자동차 사진 :", inputCarDIR)
                print("번호판 사진 :", inputFileName)
                print("광학문자 :", outputFileName)

                result = cv2.resize(result, dsize=(432, 98), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(outputFileName, result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # OCR ########################################################################################################
                ocr = out_result(outputFileName)
                resultOCR = ocr.output('1024+52font')
                print("OCR 1차 :", resultOCR)
                if bool(re.search('\d{2}\D\d{4}', resultOCR)):
                    resultOCR = re.search('\d{2}\D\d{4}', resultOCR).group()
                    print("OCR 2차 :", resultOCR)
                elif bool(re.search('\d{2}\D', resultOCR)) and bool(re.search('\d{4}', resultOCR)):
                    resultOCR = re.search('\d{2}\D', resultOCR).group() + re.search('\d{4}', resultOCR).group()
                    print("OCR 2차_2 :", resultOCR)

                txt1 = str(num) + ' : ' + str(resultOCR) + '\n'
                print(txt1)

                f.write(txt1)

                num = num + 1
                print('------------------------------------------')

                ############################################################################################################
                # response
                data = {"capture_path": capture_path, "saveEvidenceCrop": saveEvidenceCrop,
                        "saveTrackerCrop": saveTrackerCrop, "final_plate": inputFileName, "final_save": outputFileName, "txt": resultOCR}
                path_data = json.dumps(data, ensure_ascii= False)
                print(path_data)
                response = make_response(path_data)
                response.headers['Content-Type'] = 'application/json;charset=utf-8'
                print(str(response))
                return response

            else:
                print('Image Load Fail!!!', num)
                print("전체사진 :", saveEvidenceCrop)
                print("자동차 사진 :", inputCarDIR)
                txt2 = str(num) + ' : X\n'
                f.write(txt2)
                num = num + 1
                print('------------------------------------------')

                # response
                data = {"capture_path": capture_path, "saveEvidenceCrop": saveEvidenceCrop,
                        "saveTrackerCrop": saveTrackerCrop, "final_plate": inputFileName, "final_save": outputFileName,
                        "txt": "X"}
                path_data = json.dumps(data, ensure_ascii=False)
                print(path_data)
                response = make_response(path_data)
                response.headers['Content-Type'] = 'application/json;charset=utf-8'
                print(str(response))
                return response

        f.close()




if __name__ == '__main__':
    app.run(debug=True, host='localhost')


