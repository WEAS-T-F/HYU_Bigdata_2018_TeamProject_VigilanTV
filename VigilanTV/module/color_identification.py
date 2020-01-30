import cv2
import numpy as np
import imutils

# 객체를 만들 때 클래스(colorIdentification())를 호출하면서 opencv로 읽어들인 이미지 객체명을 입력함.
# color라는 함수는 사진 속 물체의 색상을 리턴함.
# 현재 빨강, 초록, 파랑, 노랑, 주황, 검은색, 하얀색/회색을 지원함.

def colorEqualize(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(image_hsv)
    V_eq = cv2.equalizeHist(V)
    image_hsv_eq = cv2.merge((H, S, V_eq))
    image_bgr_modified = cv2.cvtColor(image_hsv_eq, cv2.COLOR_HSV2BGR)
    return image_bgr_modified

def colorCluster(image, cluster = 10):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    image_clustered = res.reshape((image.shape))
    return image_clustered

def foreMask(image):
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image_masked = image*mask2[:,:,np.newaxis]
    return image_masked

def crop(image):
    H, W = image.shape[:2]
    image_crop = image[5*int(H/12):9*int(H/12),1*int(W/12):11*int(W/12)]
    image_crop = cv2.resize(image_crop,(300,300))
    return image_crop

class colorIdentification():
    def __init__(self, image):
        self.image = image

    def color(self):
        image = self.image
        image = imutils.resize(image, height=300)

        image_equalized = colorEqualize(image)
        image_blur = cv2.bilateralFilter(image_equalized, 100, 50, 50)
        # image_eq_clustered = colorCluster(image_blur)
        image_eq_zoom = crop(image_blur)

        image_hsv = cv2.cvtColor(image_eq_zoom, cv2.COLOR_BGR2HSV)

        color_list = {'red1': [0, 255, 255],
                      'orange': [13, 255, 255],
                      'yellow': [30, 255, 255],
                      'green': [60, 255, 255],
                      'sky': [90, 255, 255],
                      'blue': [120, 255, 255],
                      'purple': [150, 255, 255],
                      'red2': [180, 255, 255],
                      'black1': [0, 0, 0],
                      'black2': [0, 0, 0],
                      'white_or_gray1': [0, 0, 255],
                      'white_or_gray2': [0, 0, 255]}

        color_count = {'red': 0, 'red1': 0, 'red2': 0,
                       'orange': 0, 'yellow': 0,
                       'green': 0,
                       'sky': 0,
                       'blue': 0,
                       'purple': 0,
                       'black': 0,
                       'white_or_gray': 0,
                       'black1': 0, 'black2': 0,
                       'white_or_gray1': 0, 'white_or_gray2': 0}

        for i in list(color_list):
            if i == 'red1' or 'red2' or 'orange' or 'yellow' or 'green' or 'sky' or 'blue' or 'purple':
                lower_color = (color_list[i][0] - 11, color_list[i][1] - 210, color_list[i][2] - 210)
                upper_color = (color_list[i][0] + 10, color_list[i][1], color_list[i][2])

            if i == 'black1':
                lower_color = (5, 0, 1)
                upper_color = (35, 90, 145)

            if i == 'black2':
                lower_color = (90, 0, 1)
                upper_color = (150, 90, 145)

            if i == 'white_or_gray1':
                lower_color = (5, 0, 120)
                upper_color = (35, 70, 255)

            if i == 'white_or_gray2':
                lower_color = (95, 0, 120)
                upper_color = (130, 90, 255)

            img_mask = cv2.inRange(image_hsv, lower_color, upper_color)
            img_result = cv2.bitwise_and(image_eq_zoom, image_eq_zoom, mask=img_mask)

            array_a = img_mask.reshape((-1, 3))
            list_a = [list(array_a[j]) for j in range(len(array_a))]
            num_a = [sum(list_a[k]) for k in range(len(list_a))]
            color_count[i] = 30000 - num_a.count(0)

        color_count['red'] = color_count['red1'] + color_count['red2']
        color_count['black'] = color_count['black1'] + color_count['black2']
        color_count['white_or_gray'] = color_count['white_or_gray1'] + color_count['white_or_gray2']

        car_color = max(color_count.items(), key=lambda x: x[1])[0]

        return car_color
#
# image1 = cv2.imread('car.jpg')
# identificator = colorIdentification(image1)
# color = identificator.color()
# print(color)