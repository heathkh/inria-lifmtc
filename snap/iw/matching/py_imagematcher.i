%module py_imagematcher

%include "snap/google/base/base.swig" 
%include "std_string.i"
%include "std_vector.i"
%include "exception.i"
%include "snap/google/base/base.swig"
%include "iw/iw.swig"

%apply std::string& INPUT {const std::string&};
%apply std::string* OUTPUT {std::string*};
%apply double* OUTPUT {double*};

%{
#include "iw/matching/image_matcher.h"
%}


%include "iw/matching/image_matcher.h"



