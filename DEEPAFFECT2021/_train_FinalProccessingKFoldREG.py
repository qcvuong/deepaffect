# DATE: 06/11/21
# USAGE
# python _train_FinalProccessingKFoldREG.py --model myModel.h5 --structure myModelStructure.png --Logger myModelResults.csv --EarlyStop myModel_earlystop.h5 --plot myModelPlot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import to_categorical
# from pyimagesearch.Classifynet_Diff_Mult import ClassifyNet
# from pyimagesearch.VGG16 import vgg16
# from pyimagesearch.VGG16Scene import vgg16Scene
# from pyimagesearch.InceptronV3 import inception
# from pyimagesearch.FullnetMark2 import FullNet2
from pyimagesearch.FullnetMark3 import FullNet3
from pyimagesearch.FullnetMark3_Freeze import FullNet3FRZ
# from pyimagesearch.IndivTest import indiv
# from pyimagesearch.Peaknet import Peaknet
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
# from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from pyimagesearch.Fullnet import FullNet
import pickle
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from numpy import savetxt
import datetime
from keras.models import load_model
from pyimagesearch.VGG16 import vgg16

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-s", "--structure", type=str, required=True,
	help="path to output structure")
ap.add_argument("-t", "--Logger", type=str, required=True,
	help="path to csv logger")
ap.add_argument("-e", "--EarlyStop", type=str, required=True,
	help="path to early stop model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output accuracy/loss plot")

args = vars(ap.parse_args())


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
#NAME = (args["TB"] + datetime.datetime.now().strftime("%H%M-%d%m%Y"))
EPOCHS = 50
INIT_LR = 0.0001
BS = 15
IMAGE_DIMS = (224, 224, 3) #224, 224, 3

# initialize the model
print("[INFO] compiling model...")
model = FullNet3FRZ.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], finalAct="relu")
opt = SGD(lr=INIT_LR, momentum=0.9)
#opt = RMSprop(lr = INIT_LR)
#, decay=INIT_LR/EPOCHS
# print (model.summary())
# face = model.layers[3]
# print (face.summary())
# input()


###### Train

TRAINcsv = pd.read_csv('_train_TRAINdata.csv')

TRAINarousalLabels =  np.array(TRAINcsv.AroMean)
print(TRAINarousalLabels.shape)
# # input("key")
#
TRAINvalenceLabels = np.array(TRAINcsv.ValMean)
print(TRAINvalenceLabels.shape)
# input("key")

print("[INFO] loading images...")
train_image = []
for i in tqdm(range(TRAINcsv.shape[0])):
    img = image.load_img('full_imgs_train/'+str(TRAINcsv['FileName'][i])+'.jpg',target_size=(IMAGE_DIMS))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
TRAINimages = np.array(train_image, dtype="float") #dtype="float"


####### validation

TESTcsv = pd.read_csv('_train_VALIDATIONdata.csv')

TESTarousalLabels =  np.array(TESTcsv.AroMean)
print(TESTarousalLabels.shape)
# # input("key")
#
TESTvalenceLabels = np.array(TESTcsv.ValMean)
print(TESTvalenceLabels.shape)
# input("key")

print("[INFO] loading images...")
test_image = []
for i in tqdm(range(TESTcsv.shape[0])):
    img1 = image.load_img('full_imgs_train/'+str(TESTcsv['FileName'][i])+'.jpg',target_size=(IMAGE_DIMS))
    img1 = image.img_to_array(img1)
    img1 = img1/255
    test_image.append(img1)
TESTimages = np.array(test_image, dtype="float") #dtype="float"

######## model
#


print (model.summary())
# input()

print ("[INFO] Loading model structure...")
print(plot_model(model, to_file=(args["structure"]), show_shapes=True, show_layer_names=True))
# input("Press enter to continue")

losses = {
	"arousal_output": "mean_squared_error",
	"valence_output": "mean_squared_error",
}

model.compile(loss=losses, optimizer=opt)

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(filepath=(args["EarlyStop"]), monitor='val_loss',
                            save_best_only=True, save_weights_only=False),
			 CSVLogger(args["Logger"], separator=','),
			 ReduceLROnPlateau(monitor='val_loss', factor=0.1,
							   patience=3, min_lr=0.0000001, verbose=1)]


H = model.fit(TRAINimages, {"arousal_output": TRAINarousalLabels, "valence_output": TRAINvalenceLabels},
	batch_size=BS,
	validation_data=(TESTimages, {"arousal_output": TESTarousalLabels, "valence_output": TESTvalenceLabels}),
	epochs=EPOCHS,
	callbacks=callbacks,
	verbose=1,
	shuffle=True)



# save the model to disk
print("[INFO] saving network...")
model.save(args["model"])

print("[INFO] plotting results...")
plt.style.use("ggplot")
plt.figure()
N = EPOCHS

# ymin, ymax = 0, 6

# Set the y limits making the maximum 5% greater

plt.plot(H.history["loss"], label="Total_Train_loss")
plt.plot(H.history["valence_output_loss"], label="Train_Valence_loss")
plt.plot(H.history["arousal_output_loss"], label="Train_Arousal_loss")
plt.plot(H.history["val_loss"], label="Total_Validation_loss")
plt.plot(H.history["val_valence_output_loss"], label="Validation_Valence_loss")
plt.plot(H.history["val_arousal_output_loss"], label="Validation_Arousal_loss")
# plt.ylim(ymin, ymax)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

# print("[INFO] loading network for evaluation...")
# model = load_model(args["EarlyStop"])
#
# print("Evaluating model...")
# evaluation = model.evaluate(testX, {"arousal_output": testArousalY, "valence_output": testValenceY})
# print (model.metrics_names[2], evaluation[2])
# print (model.metrics_names[1], evaluation[1])
