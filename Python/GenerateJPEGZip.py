import numpy as n
import os
from PIL import Image
import sys
from zipfile import ZipFile
import zipfile
import cPickle as pickle


# process the folder datadir
# with subfolder names inside the id2label keys
# process only labelnum of labels
def process(id2label, datadir, labelnum, batchsize = 128):
    
    # 1. read all the images and corresponding labels under this folder
    images = []
    labels = []
    for dirs in sorted(id2label.keys()):
        label = sorted(id2label.keys()).index(dirs)
        for img in os.listdir(datadir+'/'+dirs):
            labels.append(label)
            images.append(datadir+'/'+dirs+'/'+img)
        if label >= labelnum:
            break

    # 2. randperm the images and labels
    rarray = n.random.permutation(range(len(labels)))
    num = len(labels)/batchsize
    totalnum = num
    if len(labels) % batchsize is not 0:
        totalnum = num+1

    # Split them into batches
    label_batches = []
    image_batches = []
    for i in range(num):
        sublabels = []
        filenames = []
        for idx in rarray[i*batchsize:(i+1)*batchsize]:
            sublabels.append(labels[idx])
            filenames.append(images[idx])
        label_batches.append(sublabels)
        image_batches.append(filenames)

    if totalnum > num: # still some left
        sublabels = []
        filenames = []
        for idx in rarray[num*batchsize:]:
            sublabels.append(labels[idx])
            filenames.append(images[idx])
        label_batches.append(sublabels)
        image_batches.append(filenames)

    # meta
    dic = {'batch_idx': range(totalnum), 'label_batches': label_batches, 'image_batches': image_batches}

    # zip images
    for i in dic['batch_idx']:
        zipfilename = datadir+'/'+'data_batch_'+str(i+1)
        myzip = ZipFile(zipfilename, 'w', zipfile.ZIP_STORED)
        newstaff = []
        for img in dic['image_batches'][i]:
            myzip.write(img, os.path.basename(img))
            newstaff.append(os.path.basename(img))
        dic['image_batches'][i] = newstaff
        myzip.close()

    return dic

def calcMean(datadir):
    sumlist = []
    meta = {}
    totalnum = 0
    subdirs = os.listdir(datadir)
    for files in subdirs:
        if files.startswith('data_batch_'):
            f = open(datadir+'/'+files, 'r')
            memfile = StringIO(f.read())
            f.close()
            zipf = zipfile.ZipFile(memfile, 'r', ZIP_STORED)
            # get the file list from meta
            filelist = zipf.namelist()
            totalnum += len(filelist)
            data = []
            for i in filelist:
                arr = n.array(Image.open(StringIO(zipf.read(i))))
                data.append(n.concantenate([arr[:,:,0].flatten('C'), arr[:,:,1].flatten('C'), arr[:,:,2].flatten('C')]))

            data = n.array(data)
            s = n.sum(data, axis=0)
            sumlist.append(s)

    # assert(len(sumlist)==(len(subdirs)-1))
    # assert(totalnum==1281167) # the number of imagenet training images

    sumlist = n.array(sumlist)
    allsum = n.sum(sumlist, axis=0)
    mean = n.divide(allsum, totalnum)
    mean = n.require(mean.T, n.float32, 'C')
    return mean


if __name__ == "__main__":

    # First get the ID2Label mapping.
    # map each of the foldeer to a number
    ID2Label = open(sys.argv[1]+'/'+'ID2Label')
    id2label = {}
    for line in ID2Label:
        id2label[line[:9]]=line[11:]

    # process imagenet
    labelnum = int(sys.argv[2])

    # Get the label names
    labelnames = []
    for item in sorted(id2label.keys()):
        labelnames.append(id2label[item])

    # process train and test
    train = process(id2label, sys.argv[1]+'/'+'train', labelnum)
    test  = process(id2label, sys.argv[1]+'/'+'test',  labelnum)


    # merge meta
    trainidxnum = len(train['batch_idx'])
    for i in range(len(test['batch_idx'])):
        test['batch_idx'][i] += trainidxnum

    dic = {'data_name': 'image-net', 'num_colors': 3, 'batch_size': 128, 'num_vis': 256*256, 'image_size': 256}
    dic['batch_idx'] = train['batch_idx'] + test['batch_idx']
    dic['label_batches'] = train['label_batches'] + test['label_batches']
    dic['image_batches'] = train['image_batches'] + test['image_batches']
    
    print 'training batches: 1-'+str(trainidxnum)

    # calcmean
    mean = calcMean(sys.argv[1]+'/'+'train')
    dic['data_mean'] = mean
    dic['num_cases_per_batch'] = 128
    dic['label_names'] = labelnames
    
    # dump meta
    pickle.dump(dic,open("batches.meta",'w'))

    