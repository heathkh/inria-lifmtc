%module py_featurematcher

%include "snap/google/base/base.swig" 
%include "std_string.i"
%include "std_vector.i"
%include "exception.i"
%include "snap/google/base/base.swig"
%include "iw/iw.swig"

%apply std::string& INPUT {const std::string&};
%apply std::string* OUTPUT {std::string*};

%{
#include "iw/matching/feature_matcher.h"
%}


namespace std {
   %template(vectorFeatureCorrespondence) vector<iw::FeatureCorrespondence>;
};

%apply std::vector<iw::FeatureCorrespondence>* OUTPUT {std::vector<iw::FeatureCorrespondence>*};


%include "iw/matching/feature_matcher.h"



