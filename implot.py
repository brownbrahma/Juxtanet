import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def plotimg(query, ind, impath):
	f, ax = plt.subplots(2,3)
	k = -1
	for i in range(2):
		for j in range(3):
			if(i==0 and j==0): 
				ax[i, j].imshow(mpimg.imread(query))
				ax[i, j].set_title('Query image')	
			else:
				ax[i, j].imshow(mpimg.imread(impath[ind[k]]))
				ax[i, j].set_title(str(k+1))					
			k = k+1
	plt.show()
