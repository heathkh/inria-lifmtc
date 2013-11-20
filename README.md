lifmtc
======

local image feature matching w/ geometric verification by transform clustering

Requirements
============

* CMake 2.8 or better
* CMake SNAP


Compiling
=========
```
cd <source dir> 
mkdir build
cd build
cmake ..
cmake ..   # yep... need to do it twice
make -j10  # build... parallel
 ```

Running
=========
 ```
cd <source dir>/build
./run_experiment.py
 ```
