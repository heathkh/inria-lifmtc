CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(TOMATO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${TOMATO_ROOT})

# Example of a header only C++ library package
CPP_LIBRARY(
  NAME      tomato
  HEADERS   tomato.h  Cluster_Basic.h  Core.h     Distance_ANN.h  Interval.h  Vertex.h Cluster.h        Density.h  Distance.h      Point.h  
  SOURCES   tomato.cpp
  LIB_TYPE  STATIC_AND_SHARED    
  PACKAGES  //snap/ANN:ann    
  SWIG_PY   py_tomato.i        
)           

# Example of a C++ binary that uses both a standard and header only library  
CPP_BINARY(
  NAME      cli
  SOURCES   main_w_density.cpp   
  PACKAGES  //snap/tomato:tomato                          
)           



ADD_SUBDIRECTORY(test)