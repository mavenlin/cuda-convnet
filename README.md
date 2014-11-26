NIN is now in Caffe Model Zoo
==============================
https://github.com/BVLC/caffe/wiki/Model-Zoo


cuda-convnet
============

started from Alex's code on google code


run NIN using this code
=======================

I implemented cccp (cascadable cross channel parameteric) pooling in this code.
The NIN structure is in my paper: Network In Network submitted on ICLR2014.

compiling the code
==================
To compile this code, cuda-5.0 or cuda-5.5 is required.
The other dependencies are listed [here](https://code.google.com/p/cuda-convnet/wiki/Compiling).
Setup the paths in the build.sh script under Kernel and PluginSrc.
Run ./build.sh under the main directory. A dist directory will be created with all python codes and built shared libraries inside.
change directory into dist, and add current path to LD_LIBRARY_PATH by
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
```
All the experiments are run in the dist directory.

the data
========
The datasets are preprocessed according to [maxout networks](http://arxiv.org/abs/1302.4389) using the python code in pylearn. How to preprocess the data is detailed [here](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout).
However, things are complicated because cuda-convnet has a different data format than pylearn, the way to dump cuda-convnet usable data is [here](https://code.google.com/p/cuda-convnet/wiki/Data).

The preprocessed [CIFAR-10](https://drive.google.com/file/d/0B5bEhIhshfIeNkFqS0pjeHg1Tm8/edit?usp=sharing), [CIFAR-100](https://drive.google.com/file/d/0B5bEhIhshfIeQzlrd2tEVTc3Z2M/edit?usp=sharing) and [MNIST](https://drive.google.com/file/d/0B5bEhIhshfIeeXAzS183VkhWUmM/edit?usp=sharing) datasets are available on my google drive (just follow the link), but SVHN will not as it is around 20G after preprocessing.

Here is an example code to convert the pylearn preprocessed train.pkl and test.pkl to cuda-convnet data_batch_x files for CIFAR-10 data.
```
import cPickle
import numpy
train=cPickle.load(open('train.pkl'))
for i in range(5):
    sub = train.__dict__['X'][i*10000:i*10000+10000,:]
    data=cPickle.load(open('../cifar-10-batches-py/data_batch_%d' % (i+1), 'r'))
    data['data']=numpy.require(sub.T, numpy.float32, 'C')
    cPickle.dump(data, open('data_batch_%d' % (i+1), 'w'))

test=cPickle.load(open('test.pkl'))
sub = test.__dict__['X']
data=cPickle.load(open('../cifar-10-batches-py/data_batch_6', 'r'))
data['data']=numpy.require(sub.T, numpy.float32, 'C')
cPickle.dump(data, open('data_batch_6', 'w'))
```

running the datasets
====================
CIFAR-10
--------
run the following under dist.
```shell
python convnet.py --data-path /path/to/cifar-10/pickled/data --data-provider cifar --layer-def ../NIN/cifar-10_def --layer-params ../NIN/cifar_10-params --train-range 1-5 --test-range 6 --save-path /path/to/save/the/model/ --test-freq 20 --epochs 200
```
This trains the model defined in NIN/cifar-10_def using the the parameter in NIN/cifar-10_params for 200 epochs.
After 200 epochs, the test error is around 14%
After this, edit the NIN/cifar-10_params file by changing all the epsW to one tenth of the original value.
and run:
```shell
python convnet.py -f /path/where/model/was/saved/ --epochs 230
```
This will run the model with the adjusted parameters for another 30 epochs, which results in error rate near 10.7%

Then change the epsW to one tenth again and run:
```shell
python convnet.py -f /path/where/model/was/saved/ --epochs 260
```
Then the error rate will be 10.4%

CIFAR-100
---------
CIFAR-100 is similar to CIFAR-10 but just replace some parameters in the script.
```shell
python convnet.py --data-path /path/to/cifar-100/pickled/data --data-provider cifar --layer-def ../NIN/cifar-100_def --layer-params ../NIN/cifar_100-params --train-range 1-5 --test-range 6 --save-path /path/to/save/the/model/ --test-freq 20 --epochs 200
```
The rest is the same with CIFAR-10


SVHN
----
Coming soon.


MNIST
-----
Coming soon.
