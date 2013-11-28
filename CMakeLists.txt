CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
FIND_PACKAGE(cmakesnap) # enables CMakeSNAP support in all subdirectories

ADD_SUBDIRECTORY(snap)

PY_BINARY(
  NAME      run_experiment
  SOURCES   run_experiment.py   
  PACKAGES  //snap/tomato:tomato    
            //snap/iw/matching:py_featureextractor
            //snap/iw/matching:py_featurematcher  
  DATA      exp_result_template.html                       
)           

CHECK_NO_MISSING_PACKAGES()
 