from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy
from sklearn.manifold import TSNE

def plots(x2):
	plt.scatter(x2[:,0], x2[:,1])
	plt.show()

def vPca(filepath):
	df = pd.read_csv(filepath_or_buffer=filepath, sep=',')
	x = df.ix[:,:].values
	x = x.transpose()
	sklearn_pca = sklearnPCA(n_components=2)
	x2 = sklearn_pca.fit_transform(x)
	plots(x2)
	
def vTsne(filepath):	
	df = pd.read_csv(filepath_or_buffer=filepath, sep=',')
	x = df.ix[:,:].values
	x = x.transpose()
	sklearn_pca = sklearnPCA(n_components=50)
	x50 = sklearn_pca.fit_transform(x)	
	tsne = TSNE(n_components=2,n_iter=1000)
	x2 = tsne.fit_transform(x50)
	plots(x2)
