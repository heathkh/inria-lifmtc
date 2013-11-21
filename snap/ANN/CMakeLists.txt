CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(ANN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
INCLUDE_DIRECTORIES(${ANN_ROOT})

CPP_LIBRARY(
  NAME      ann
  HEADERS   ANN.h      ANNx.h     kd_fix_rad_search.h  kd_search.h  kd_tree.h  pr_queue.h  ANNperf.h  bd_tree.h  kd_pr_search.h       kd_split.h   kd_util.h  pr_queue_k.h     
  SOURCES   ANN.cpp                bd_tree.cpp            kd_pr_search.cpp  kd_util.cpp bd_fix_rad_search.cpp  brute.cpp              kd_search.cpp     perf.cpp bd_pr_search.cpp       kd_dump.cpp            kd_split.cpp bd_search.cpp          kd_fix_rad_search.cpp  kd_tree.cpp  
  LIB_TYPE  STATIC
)           


