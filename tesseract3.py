import cv2
from cv2 import bitwise_and
import numpy as np
import pytesseract
import imutils
from pytesseract import Output 

img = cv2.imread("22.jpg")

gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtre = cv2.bilateralFilter(img, 7,200,200)
kose = cv2.Canny(filtre,40,200)
hImg, wImg, _ = img.shape

kontur,a = cv2.findContours(kose,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = imutils.grab_contours((kontur,a))
cnt = sorted(cnt,key=cv2.contourArea,reverse=True)[:10]

ekran = 0

for i in cnt:
    eps = 0.018*cv2.arcLength(i,True)
    aprx = cv2.approxPolyDP(i,eps,True)
    if len(aprx)==4:
        ekran = aprx
        break
    
maske = np.zeros(gri.shape,np.uint8)
yenimaske = cv2.drawContours(maske,[ekran],0,(255,255,255),-1)
    
yazi = bitwise_and(img,img,mask=maske)
    
(x,y)=np.where(maske==255)
(ustx,usty)=(np.min(x),np.min(y))
(altx,alty)=(np.max(x),max(y))
kirp = gri[ustx:altx+1,usty:alty+1]

###BINARY DONUSUMU VE KARAKTER KÜMELEMESİ

kirp = cv2.adaptiveThreshold(kirp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,95,6)


data = pytesseract.image_to_data(kirp, output_type=Output.DICT)
c = pytesseract.image_to_boxes(kirp)

print(data)
print(c)

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(kirp,lang="eng",config=custom_config)


n_boxes = len(data['text'])
for i in range(n_boxes):
    if float(data['conf'][i]) > 0:
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        kirp = cv2.rectangle(kirp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
print("************************************** \n")        
print("Tespit edilen plaka: " + text)
print("**************************************")


cv2.imshow("orjinal0",img)
cv2.imshow("kesilmis",kirp)

cv2.waitKey(0)
cv2.destroyAllWindows()

