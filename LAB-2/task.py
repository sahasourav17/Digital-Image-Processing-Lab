from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

filepath = 'Lenna.png'
img = cv.imread(filepath)
h,w,c = img.shape
print(type(img))
print(img.shape)
plt.imshow(img)
plt.xlabel('Original image in BGR')
plt.savefig('BGR_image.jpg')
plt.show()

# %%
#COLOR TO GRAYSCALE
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

print(type(img))
print(img.shape)
plt.imshow(img, cmap= 'gray')
plt.xlabel('Original image in grayscale')
plt.savefig('Grayscale.jpg')
plt.show()

# %%
#ROTATE 90 DEGREE CCW
img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
print(type(img))
print(img.shape)
plt.imshow(img, cmap= 'gray')
plt.xlabel('90 degree CCW rotated image')
plt.savefig('90_CCW.jpg')
plt.show()

# %%
#ARBITRARY ANGLE ROTATE
mat = cv.getRotationMatrix2D((h/2, w/2), 45, 1)
img = cv.warpAffine(img,mat,(h,w), borderValue=100 )
print(type(img))
print(img.shape)
plt.imshow(img, cmap= 'gray')
plt.xlabel('45 degree rotated image')
plt.savefig('arbitrary rotated.jpg')
plt.show()

# %%
#IMAGE TRANSLATION
tx = w/4; ty= h/4
mat = np.array([ [1,0,tx], [0,1,ty] ], dtype=np.float32)
img = cv.warpAffine(img, mat, (h,w))
print(type(img))
print(img.shape)
plt.imshow(img, cmap= 'gray')
plt.xlabel('Translated Image')
plt.savefig('Translated.jpg')
plt.show()

#IMAGE BINARIZATION USING BUILT IN OTSU THRESHOLDING
img = cv.imread(filepath)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
val, img = cv.threshold(img, 154,255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(val)
plt.imshow(img, cmap= 'gray')
plt.xlabel('Binary image using built-in OTSU method')
plt.savefig('Binary image.jpg')
plt.show()