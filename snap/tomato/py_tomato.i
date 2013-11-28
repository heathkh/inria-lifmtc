%module py_tomato
 
%include "exception.i"
%include "std_vector.i"

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%{
#include "snap/tomato/tomato.h"
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* data, int nb_points, int point_dim)};

class PersistanceDiagramPoint {
public:
  PersistanceDiagramPoint(){}
  PersistanceDiagramPoint(double birth, double death) :
    birth(birth),
    death(death) {
  }
  double birth;
  double death;
};

class Cluster {
public:
  Cluster(){}
  Cluster(double birth, double death, double persistence) :
    birth(birth),
    death(death),
    persistence(persistence){
  }
	double birth;
	double death;
	double persistence;
	std::vector<int> member_indices;
};


%extend Cluster { 
%pythoncode { 
    def __str__(self): 
        #members = [v for v in self.member_indices]
        return 'persistence: %2.1f  birth: %2.1f death: %2.1f num members: %s' % (self.persistence, self.birth, self.death, self.member_indices.size()) 
   } 
} 

namespace std {
   %template(vectori) vector<int>;
   %template(vectorpersistencepoints) vector<PersistanceDiagramPoint>;   
   %template(vectorcluster) vector<Cluster>;
};

%apply std::vector<PersistanceDiagramPoint>* OUTPUT {std::vector<PersistanceDiagramPoint>*};
%apply std::vector<Cluster>* OUTPUT {std::vector<Cluster>*};

%include "snap/tomato/tomato.h"

