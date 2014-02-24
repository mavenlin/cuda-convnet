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


CIFAR-10
========
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
