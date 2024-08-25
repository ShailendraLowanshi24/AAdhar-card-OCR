import cv2
import numpy as np
import pytesseract
import re
import json
from deepface import DeepFace

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Guest-Pvips\PycharmProjects\pythonProject\Aadhar-Card-OCR-master\tesseract.exe'

img3 = cv2.imread('testdata/a_0_8120.jpg')   #any picture

# removing shadow/noise from image which can be taken from phone camera

rgb_planes = cv2.split(img3)
result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))        #change the value of (10,10) to see different results
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)
dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)

text = pytesseract.image_to_string(dst).upper().replace(" ", "")

name=str

# date = str(re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)).replace("]", "").replace("[","").replace("'", "")
# print(date)
number = str(re.findall(r"[0-9]{11,12}", text)).replace("]", "").replace("[","").replace("'", "")
print(number)
#
# out_file = open("myfile.json", "w")
#
# json.dump(number, out_file, indent=6)
#
# # out_file.close()
# ()

# sex = str(re.findall(r"MALE|FEMALE", text)).replace("[","").replace("'", "").replace("]", "")
# print(sex)

cv2.imshow('original',img3)
cv2.imshow('edited',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

net = cv2.dnn.readNetFromCaffe(r"C:\Users\Guest-Pvips\PycharmProjects\pythonProject\Aadhar-Card-OCR-master\Face-Detection-master\weights-prototxt.txt", r"C:\Users\Guest-Pvips\PycharmProjects\pythonProject\Aadhar-Card-OCR-master\Face-Detection-master\res_ssd_300Dim.caffeModel")

# load the input image by resizing to 300x300 dims
# image = cv2.imread(img3)
(height, width) = img3.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img3, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

# pass the blob into the network
net.setInput(blob)
detections = net.forward()

# loop over the detections to extract specific confidence
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # greater than the minimum confidence
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x1, y1, x2, y2) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2 - y1) + ", " + str(x2 - x1) + " )"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(img3, (x1, y1), (x2, y2),
                      (0, 0, 255), 2)
        cv2.putText(img3, text, (x1, y),
                    cv2.LINE_AA, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", img3)
cv2.waitKey(0)