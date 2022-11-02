
# python test_network.py --model s.model --launch_directory images/examples/*.jpg

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument('-l', '--launch_directory', required=True,
	help="path to input image")
args = vars(ap.parse_args())

cv_img = []
i=0
for img in glob.glob(args["launch_directory"]):
    n= cv2.imread(img)
    cv_img.append(n)
for image in cv_img:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig = image.copy()

    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] loading network...")
    model = load_model(args["model"])

    (scissors, knife) = model.predict(image)[0]

    label = "scissors" if scissors > knife else "knife"
    proba = scissors if scissors > knife else knife	
    label = "{}: {:.2f}%".format(label, proba * 100)

    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

    cv2.imwrite('final images/final images'+ str(i) + ".jpg",output)
    i=i+1
    cv2.waitKey(0)
