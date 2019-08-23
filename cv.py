import numpy as np
import math
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import urllib.request

#Image processing
model=load_model('neural.model')
image= cv2.imread("cardboard13.jpg")

image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# classify the input image
(nonbio, bio) = model.predict(image)[0]
# build the label
label = "bio" if bio > nonbio else "nonbio"
y=label

print(y)

