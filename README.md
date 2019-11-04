# Juxtanet
A simple implementation of image ranking using Keras.

-----------------------------------------------------------------

The "images" folder is a dump of all images available in our
dataset (note: all  classes are mixed).

Given the path of such a dump-directory (here, images)
"extract.py" outputs "fi.csv" file which contains features
of all images with image-IDs as a header.

"juxtanet.py" can be used to:
1)Visualize these features in a 2-dimensional space.
2)Extract images similar to a give query image.

Running "juxtanet.py" you will output 5 similar
images for the given query image.
