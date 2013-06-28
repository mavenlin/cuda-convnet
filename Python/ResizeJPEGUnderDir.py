import numpy
import Image
import os
import os.path
import sys
import math


def get256x256Image(imagename):
	im = Image.open(imagename)
	width, height = im.size
	I = []
	vec = []
	if width < height:
		# Resizing image
		newheight = math.floor(256*float(height)/float(width))
		# print 'newheight '+str(newheight)
		if newheight%2 != 0:
			newheight = newheight+1
		# print 'newheight '+str(newheight)
		im = im.resize((256,int(newheight)),Image.ANTIALIAS)
		
		# Crop Image to 256 x 256
		I = numpy.asarray(im)
		if I.ndim < 2:
			print 'error ' + imagename
		elif I.ndim == 2:
			margin = int((newheight-256)/2)
			newheight = int(newheight)
			I = I[margin:newheight-margin,:]
		else:
			margin = int((newheight-256)/2)
			newheight = int(newheight)
			I = I[margin:newheight-margin,:,:]
	else:
		# Resizing image
 		newwidth = math.floor(256*float(width)/float(height))
		# print 'newwidth '+str(newwidth)
		if newwidth%2 != 0:
			newwidth = newwidth + 1
		# print 'newwidth '+str(newwidth)
		im = im.resize((int(newwidth),256),Image.ANTIALIAS)			
		
		# Crop Image to 256 x 256
		I = numpy.asarray(im)
		if I.ndim < 2:
			print 'error ' + imagename
		elif I.ndim == 2:
			margin = int((newwidth-256)/2)
			newwidth = int(newwidth)
			I = I[:,margin:newwidth-margin]
		else:
			margin = int((newwidth-256)/2)
			newwidth = int(newwidth)
			I = I[:,margin:newwidth-margin,:]
	return I



if __name__ == "__main__":

	subdirs = os.listdir(sys.argv[1])
	currdir = sys.argv[1]
	another = sys.argv[2]

	if os.path.exists(another):
		pass
	else:
		os.mkdir(another)

	for dirs in subdirs:
		if os.path.exists(os.path.join(another, dirs)):
			pass
		else:
			os.mkdir(os.path.join(another, dirs))

		for img in os.listdir(sys.argv[1]+'/'+dirs):
			filename = sys.argv[1]+'/'+dirs+'/'+img
			I = get256x256Image(filename)
			jpeg = Image.fromarray(I.astype(n.uint8))
	    	jpeg.save(filename.replace(currdir, another))
