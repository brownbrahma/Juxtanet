from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import sys, glob, os
import numpy as np
from tqdm import tqdm
import pandas as pd

#setting up the NN model
model = ResNet50(weights='imagenet', include_top=False)

#extracting features from the model
def featureExtraction(x):
	features = model.predict(x)
	f = features.flatten()
	return f

def getFeaturesFromFile(filename):
	img = image.load_img(filename, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	f = featureExtraction(x)
	return f

def getFeaturesFromDir(dirName):
	types = ('*.jpg', '*.JPEG', '*.png')
	imageFilesList = []
	for files in types:
		imageFilesList.extend(glob.glob(os.path.join(dirName, files)))
	
	imageFilesList = sorted(imageFilesList)
	features = [];
	for imfile in tqdm(enumerate(imageFilesList),desc='images'):
		f = getFeaturesFromFile(imfile[1])
		features.append(f)
	return (features, imageFilesList)

ParentDIR = 'images'
fea = getFeaturesFromDir(ParentDIR)
features = np.asarray(fea[0])
features = features.transpose()
images = fea[1]

df = pd.DataFrame(features)
df.to_csv("fi.csv", header=images,index=False)

