CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


SET(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
INCLUDE_DIRECTORIES(${ROOT})



PROTO_LIBRARY(
  NAME      proto
  PROTO     iw.proto
)


CPP_LIBRARY(
  NAME      util
  SOURCES   util.cpp
  HEADERS   util.h            
  PACKAGES  //snap/google/base:base
            //snap/google/glog:glog
            //snap/iw:proto
  LIB_TYPE  STATIC  
)

ADD_SUBDIRECTORY(matching)
