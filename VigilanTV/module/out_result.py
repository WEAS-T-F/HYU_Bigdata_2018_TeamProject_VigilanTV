import pytesseract
import PIL

class out_result :
    def __init__(self, img_path):
        self.img_path = img_path

    def output(self, ocr_lang):
        fp = open(self.img_path, "rb")
        img = PIL.Image.open(fp)
        config = ('-l ' + ocr_lang + ' --oem 0 --psm 6')
        txt = pytesseract.image_to_string(img, config= config)
        print("ocr 결과 : " + txt)
        return txt

# ocr = out_result('img/rcar1.jpg')
# resultOCR = ocr.output()