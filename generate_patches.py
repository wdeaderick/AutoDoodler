import os.path
from PIL import Image
import numpy
import math
from sklearn.feature_extraction import image
from sklearn.preprocessing import normalize
import scipy.misc

original_files = []
## Get filenames
for filename in os.listdir():
    original_files.append(filename)

## Extract patches and save them
for filename in original_files:    
    if filename[-3:] == "jpg" or filename[-3:] == "png":
        im = Image.open(filename).convert('L')
        (width, height) = im.size ##PIL convention width first
        greyscale_map = list(im.getdata())
        greyscale_map = numpy.array(greyscale_map)
        greyscale_map = greyscale_map.reshape((height, width)) ##numpy height first
        patches = image.extract_patches_2d(greyscale_map, (256, 256), max_patches = 500)
        for image_ind in range(500):
            curimage = patches[image_ind,:,:]
            curimage = scipy.misc.imresize(curimage,size = (64,64)) 
            curimage = normalize(curimage)
            name = filename[:-4] + "patch" + str(image_ind) + ".jpg"
            scipy.misc.imsave(name, curimage)
