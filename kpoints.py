from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy

def kneigh(query, features):
	df = pd.read_csv(filepath_or_buffer=features, sep=',')
	x = df.ix[:,:].values
	x = x.transpose()
	neigh = NearestNeighbors(n_neighbors=5)	
	neigh.fit(x)
	kIndices = neigh.kneighbors(query, return_distance=False)
	return kIndices
