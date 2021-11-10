# DATE: 06/11/21
# USAGE
# python _test_Prediction.py --model myModel.h5


import matplotlib
matplotlib.use("Agg")
import keras
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
# from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import LabelBinarizer
# from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
# from __builds__.Classifynet_Diff_Mult import ClassifyNet
# from __builds__.VGG16 import vgg16
# from __builds__.VGG16Scene import vgg16Scene
# from __builds__.InceptronV3 import inception
import pickle
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from numpy import savetxt
from keras.models import load_model

from keract import get_activations, display_activations, display_heatmaps

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to input model")
# ap.add_argument("-t", "--Logger", type=str, required=True,
# 	help="path to csv logger")
args = vars(ap.parse_args())


IMAGE_DIMS = (224, 224, 3)


print("[INFO] loading network...")
model = load_model(args["model"])


# initialize the data and labels

test = pd.read_csv('_test_data.csv')
ArousalScore = np.array(test.AroMean) #AroMean

ValenceScore = np.array(test.ValMean)

labels =[]
for i in tqdm(range(test.shape[0])):
    image_labels = str(test['FileName'][i])
    labels.append(image_labels)

# print(labels)
# input()

print("[INFO] loading images...")
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('full_imgs_test/'+str(test['FileName'][i])+'.jpg',target_size=(IMAGE_DIMS))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
images = np.array(test_image, dtype="float") #dtype="float"


# print (model.metrics_names[1], predict[1]) #first one is valence
# print (model.metrics_names[2], predict[2])


import csv
with open("aaPredictions/PREDICTIONS.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)

    print("predicting model scores...")
    predict = model.predict(images, verbose=1)
    for prediction in predict:
        print(type(prediction))
        for i in prediction:
            writer.writerow(i)  # pass an array of values to write a row to the csv
            print(i)
        writer.writerow("-- -- -- --")

with open("aaPredictions/AROUSAL.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    print("AROSUAL LABELS")
    for i in ArousalScore:
        print(i)
        writer.writerow([i])

with open("aaPredictions/VALENCE.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    print("VALENCE LABELS")
    for i in ValenceScore:
        print(i)
        writer.writerow([i])

with open("aaPredictions/LABELS.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    print("Labels")
    for i in labels:
        print(i)
        writer.writerow([i])


# print("Evaluating model...")
# evaluation = model.evaluate(images, {"arousal_output": ArousalScore, "valence_output": ValenceScore})
# print (model.metrics_names[2], evaluation[2])
# print (model.metrics_names[1], evaluation[1])
