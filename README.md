## HYU_Bigdata_2018_TeamProject_VigilanTV

<br>

### <p align="center">딥러닝 기반 어린이보호구역 불법 주·정차 자동탐지 시스템<p/>
# **<p align="center">VIGILANTV</p>**

## **Explanation**
과학기술정보통신부, IITP에서 지원하고 한양대학교에서 주관한 "프로젝트 주도형 빅데이터 전문가 양성과정"에서 진행한 산학협력 프로젝트입니다.

## **Summary**
![slide1](./images/Presentation_material/slide1.png)
![slide2](./images/Presentation_material/slide2.png)
![slide3](./images/Presentation_material/slide3.png)
![slide4](./images/Presentation_material/slide4.png)
![slide5](./images/Presentation_material/slide5.png)
![slide6](./images/Presentation_material/slide6.png)
![slide7](./images/Presentation_material/slide7.png)
![slide8](./images/Presentation_material/slide8.png)
![slide9](./images/Presentation_material/slide9.png)
![slide10](./images/Presentation_material/slide10.png)
![slide11](./images/Presentation_material/slide11.png)
![slide12](./images/Presentation_material/slide12.png)
![slide13](./images/Presentation_material/slide13.png)
![slide14](./images/Presentation_material/slide14.png)
![slide15](./images/Presentation_material/slide15.png)
![slide16](./images/Presentation_material/slide16.png)
![slide17](./images/Presentation_material/slide17.png)
![slide18](./images/Presentation_material/slide18.png)
![slide19](./images/Presentation_material/slide19.png)
![slide20](./images/Presentation_material/slide20.png)
![slide21](./images/Presentation_material/slide21.png)
![slide22](./images/Presentation_material/slide22.png)
![slide23](./images/Presentation_material/slide23.png)
![slide24](./images/Presentation_material/slide24.png)
![slide25](./images/Presentation_material/slide25.png)
![slide26](./images/Presentation_material/slide26.png)
![slide27](./images/Presentation_material/slide27.png)
![slide28](./images/Presentation_material/slide28.png)
![slide29](./images/Presentation_material/slide29.png)
![slide30](./images/Presentation_material/slide30.png)
![slide31](./images/Presentation_material/slide31.png)
![slide32](./images/Presentation_material/slide32.png)
![slide33](./images/Presentation_material/slide33.png)
![slide34](./images/Presentation_material/slide34.png)
![slide35](./images/Presentation_material/slide35.png)

<br>

## **What roles I took**
1. **번호판 탐지를 위한 Yolo 모델 학습**

    LabelImg를 이용해 차량 이미지에서 번호판을 라벨링 한 후,\
    AWS의 GPU 서버를 활용해 Yolo에 쓰일 weight파일을 학습시킴.

<br>

2. **객체 탐지 및 추적 알고리즘 적용**

    ![image](./images/tracking.gif)

    - Darkflow : 오픈소스 객체 탐지 알고리즘 Yolo의 신경망 Darknet을 tensorflow로 구현한 것.

    - Sort 알고리즘 : 추적 알고리즘. Darkflow로 탐지한 객체를 연속적으로 추적하기 위해서 사용

<br>

3. **차량 색상 판별 모듈 제작**

    ![image](./images/preprocessing.png)

    HSV 색상으로 전환한 이미지를 OpenCV의 inRange 메서드를 이용해 다양한 색상으로 필터링한다(이때, 일치하는 부분은 흰색, 일치하지 않는 부분은 검은색으로 나타난다). 차량의 색상은 그렇게 필터링한 색상들 중 가장 많은 흰색 픽셀 수가 검출된 색상이 된다.

    HSV 색상을 필터링하는데 있어 시간대/날씨/그림자 여부 등으로 인한 명도 변화, 창문/타이어/아스팔트 등 차량 색상과 관계없는 요소 등에 따라 결과의 오차가 커질 수 있기 때문에, 다음과 같은 이미지 전처리가 선행되어야 한다. 

    - **크기조절 / Resizing**
        
        어떤 이미지를 처리하든 동일한 크기로 처리하게 해야 이후의 작업을 처리하기 용이하다. 이 모듈의 경우 이미지의 높이가 300px이 되도록 조정했다.

    - **히스토그램 평활화 / Histogram Equalization**

        이미지의 명도를 균일화해 시간대/날씨/공간에 따른 그림자 여부 등에 따른 명도차를 줄일 수 있다. 다시 말해, 이 작업을 통해 어두운 곳에서 찍은 파란색 자동차가 검은색으로 인식될 확률을 줄이는 것이 가능하다.

    - **필터 / Filter**

        명도가 낮은 곳에서 찍힌 사진과 같은 경우 이미지 내에 노이즈가 많이 발생한다. 이 때 필터를 적용하면 노이즈를 줄여 더 정확한 색 검출이 가능하다. 본 모듈에서는 Bilateral 필터를 적용했다.

    - **잘라내기 / Cropping**

        아스팔트, 타이어, 창문 등 본 차체의 색상과 무관한 주변 환경의 색상으로부터 영향을 줄이기 위해 적용한다. 이 과정을 거칠 경우, 이 과정을 거치지 않았을 때보다 검은색이 필요 이상으로 검출되어 전혀 다른 색이 검은색으로 판별되는 경우가 현저하게 줄어들었다.\
        사진을 잘라낼 때는 차체의 색상과 무관한 부분을 피해 사진의 중간부 하단을 일정 부분 잘라낸다.

<br>

4. **빅데이터플랫폼 구축**

    ![image](./images/database.png)
    
    본 프로젝트는 프로세스의 결과물인 텍스트, 이미지 데이터를 빅데이터플랫폼인 하둡 에코 시스템에 저장했다. 본 시스템의 세부 사항은 다음과 같다.

    - 노드 갯수 : 3
    - 가상머신 : VMWare
    - 운영체제 : Linux CentOS 7

    본 프로젝트의 목적은 하둡 에코 시스템의 구축이 아니었기 때문에, 시간관계상 데이터를 저장할 수 있는 최소한의 요건으로 시스템을 구축했다. 

    - **Apache NiFi**
    
        데이터를 실시간으로 수집하고 저장소에 저장하는 역할로 사용했다.\
        단속 날짜, 차량 번호 등 텍스트 데이터의 경우 JSON의 형태로 http request를 통해 받아 Apache Hbase에 저장했으며,\
        차량 이미지, 번호판 이미지 등 이미지 데이터의 경우 ftp 서버에 저장한 이미지 파일을 실시간으로 추출해 HDFS에 저장했다.

    - **Cloudera manager**

        Hadoop 시스템의 설치와 관리를 용이하게 할 매니지먼트 프로그램으로써 사용. 버전은 5.16.1
    
    - **Apache HBase**

        NiFi를 통해 텍스트 데이터를 받아 저장하는데 사용.

    - **Apache Hive**

        HBase의 데이터를 HiveQL을 통해 관리하고 웹과 연동하기 위해 사용.

    - **Hue**

        웹과 연동할 HiveQL 쿼리를 작성하기 위해 사용


