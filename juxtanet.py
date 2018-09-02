from keras.applications.resnet50 import ResNet50
from visualise import vPca, vTsne
from kpoints import kneigh
from implot import plotimg
import numpy
import pandas as pd
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

Ffilepath = 'fi.csv'

#setting up the NN model
model = ResNet50(weights='imagenet', include_top=False)

#extracting image IDs
df = pd.read_csv(filepath_or_buffer=Ffilepath, sep=',')
impath = list(df)

#visualise extracted features
#vPca(Ffilepath)
#vTsne(Ffilepath)

#find nearest neighbours
queryIm = 'k.JPEG'

img = image.load_img(queryIm, target_size=(224, 224))
qi = image.img_to_array(img)
qi = numpy.expand_dims(qi, axis=0)
qi = preprocess_input(qi)
q = model.predict(qi)
q = numpy.reshape(q.flatten(),(-1,2048))

k = kneigh(q,Ffilepath)
k = k.flatten()
indices = k.tolist()

#plot images
plotimg(queryIm, indices, impath)

