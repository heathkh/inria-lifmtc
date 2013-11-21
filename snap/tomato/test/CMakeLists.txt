CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

  
PY_BINARY(
  NAME      test
  SOURCES   test.py   
  PACKAGES  //snap/tomato:tomato     
  DATA      spiral_w_o_density.txt       
            toy_example_w_o_density.txt              
)           



