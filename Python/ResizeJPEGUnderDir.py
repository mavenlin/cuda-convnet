import numpy
import Image
import os
import os.path
import sys
import math


def get256x256Image(imagename):
	im = Image.open(imagename)
	im = im.convert('RGB')
	width, height = im.size
	if width < height:
		# Resizing image
		newheight = math.floor(256*float(height)/float(width))
		# print 'newheight '+str(newheight)
		if newheight%2 != 0:
			newheight = newheight+1

		margin = int((newheight-256)/2)
		newheight = int(newheight)
		im = im.resize((256,int(newheight)),Image.ANTIALIAS)

		# Crop Image to 256 x 256
		return im.crop((0, margin, 256, newheight-margin))
	else:
		# Resizing image
 		newwidth = math.floor(256*float(width)/float(height))
		# print 'newwidth '+str(newwidth)
		if newwidth%2 != 0:
			newwidth = newwidth + 1

		margin = int((newwidth-256)/2)
		newwidth = int(newwidth)
		im = im.resize((int(newwidth),256),Image.ANTIALIAS)	
		
		# Crop Image to 256 x 256
		return im.crop((margin, 0, newwidth-margin, 256))



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
			print 'making dir '+os.path.join(another, dirs)
			os.mkdir(os.path.join(another, dirs))

		for img in os.listdir(os.path.join(currdir, dirs)):
			filename = os.path.join(currdir, dirs, img)
			im = get256x256Image(filename)
			assert(im.size == (256,256))
			im.save(filename.replace(currdir, another))
