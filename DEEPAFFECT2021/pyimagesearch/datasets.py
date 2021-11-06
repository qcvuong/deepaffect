# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from imutils import paths
import random
from keras.preprocessing.image import img_to_array

def load_score_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["Score"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	print(df)
	# return the data frame
	return df

	#unsure how the arousal score attributes link to the images

def load_arousal_images(df, inputPath):
	# initialize our images array (i.e. the images themselves)
	data = []

	imagePaths = sorted(list(paths.list_images(inputPath)))
	#print(imagePaths)

	#random.seed(42)
	#random.shuffle(imagePaths)

	# loop over the input images
	for index in range(1, len(imagePaths)+1):
		imagePath = inputPath + "\\" + str(index) + ".jpg"


	#for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (224, 224)) #64, 64, 3
		image = img_to_array(image)
		data.append(image)

		print(imagePath)

	# return our set of images
	return np.array(data)

