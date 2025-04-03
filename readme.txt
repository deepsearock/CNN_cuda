Lab3: CNN

Zhaorui Wang
cwid: 20007447

The lab and spreadsheet with some data 

In order to build you must run

mkdir build && cd build
cmake ..
cmake --build .

This will yield 2 programs
./comparison
./graph

./comparison utilization:
Usage: ./comparison -i <dimX> -j <dimY> -k <dimK> 
  this compares optimized (shared and textured memory), shared(global and shared), naive (global), and cpu cnn functions

./graph -i <dimX> -j <dimY> 
  this program does the same as the top but runs through 4-20 masks and outputs the results in to a file. 