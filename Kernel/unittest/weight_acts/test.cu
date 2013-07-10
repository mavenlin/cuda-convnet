#include "matrix.h"
#include "nvmatrix.cuh"
#include <iostream>
#include "cudaconv2.cuh"

using namespace std;

int main(int argc, char ** argv)
{
   ifstream fin;
   cout<<"1"<<endl;
   fin.open (argv[1]);
   if ( !fin ) exit( 1 );
   cout<<"2"<<endl;
   int imgSize, imgNum, imgColor, FilterSize, FilterNum, ActSize; // Sizes needed for array
   float _imgSize, _imgNum, _imgColor, _FilterSize, _FilterNum, _ActSize; // Sizes needed for array
   
   fin >> _imgSize;
   fin >> _imgNum;
   fin >> _imgColor;
   fin >> _FilterSize;
   fin >> _FilterNum;
   fin >> _ActSize;

   imgSize = (int)_imgSize;
   imgNum = (int)_imgNum; 
   imgColor = (int)_imgColor;
   FilterSize = (int)_FilterSize;
   FilterNum=(int)_FilterNum;
   ActSize = (int)_ActSize; // Sizes needed for array

   cout<<"imgSize"<<imgSize<<"imgNum"<<imgNum<<"imgColor"<<imgColor<<"FilterSize"<<FilterSize<<"FilterNum"<<FilterNum<<"ActSize"<<ActSize<<endl;
   cout<<"3"<<endl;
   float * _img = new float[imgNum*imgColor*imgSize*imgSize];
   float * _fil = new float[FilterSize*FilterSize*FilterNum*imgColor];
   float * _act = new float[imgNum*ActSize*ActSize*FilterNum];
 
   cout<<"4"<<endl;
   for(int i=0; i<imgNum*imgColor*imgSize*imgSize; i++)
      fin >> _img[i];
   for(int i=0; i<FilterSize*FilterSize*FilterNum*imgColor; i++)
      fin >> _fil[i];
   for(int i=0; i<imgNum*ActSize*ActSize*FilterNum; i++)
      fin >> _act[i];
   fin.close();

   float * result = new float[FilterSize*FilterSize*FilterNum*imgColor];
   float * presult = new float[FilterSize*FilterSize*FilterNum*imgColor*ActSize*ActSize];

   Matrix inp(_img, imgColor*imgSize*imgSize, imgNum);
   Matrix fil(_fil, FilterSize*FilterSize*imgColor, FilterNum);
   Matrix res(result, FilterSize*FilterSize*imgColor, FilterNum);
   Matrix out(_act, ActSize*ActSize*FilterNum, imgNum);
   Matrix pre(presult, ActSize*ActSize*FilterSize*FilterSize*imgColor, FilterNum);

   NVMatrix pr(pre, true);
   NVMatrix re(res, true);
   NVMatrix in(inp, true);
   NVMatrix fi(fil, true);
   NVMatrix ou(out, true);
   
   convWeightActs(in, ou, pr, imgSize, ActSize, ActSize, FilterSize, 0, 1, imgColor, 1, 1);
   
   pr.reshape(ActSize*ActSize, FilterSize*FilterSize*FilterNum*imgColor);
   NVMatrix& summed = pr.sum(0);
   summed.reshape(imgColor*FilterSize*FilterSize, FilterNum);
   // convImgActs(ou, fi, re, imgSize, imgSize, ActSize, 0, 1, imgColor, 1);
   summed.copyToHost(res);
   res.subtract(fil);
   res.eltWiseDivide(fil);
   float max_ = res.max();
   float min_ = res.min();
   res.print();
   cout<<"max:"<<max_<<"min:"<<min_<<endl;
   // inp.print();

}
