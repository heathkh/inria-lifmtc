CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

CPP_BINARY (
  NAME      vgg_test
  SOURCES   vgg_test.cpp  
  PACKAGES  //snap/iw/matching:featureextractor                        
)




