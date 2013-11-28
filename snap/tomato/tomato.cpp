#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <limits>

#include "Vertex.h"
#include "Distance_ANN.h"
#include "Core.h"
#include "Cluster_Advanced.h"
#include "Point.h"
#include "Density.h"

using namespace std;

//rename for brevity
typedef Vertex<ANNPoint,Cluster_Info > Point;

// comparison function object for vector indices
template<class V> class Less_Than {
  protected:
  V& v;
  public:
  Less_Than (V& v_): v(v_){}
  bool operator()(const int a, const int  b) const 
  {return Point::Less_Than()(v[a], v[b]);}
};


void cluster_base(double* data, int nb_points, int point_dim, int num_neighb, double r, double persistence_threshold, std::vector<Cluster>* clusters, std::vector<PersistanceDiagramPoint>* diagram){
  vector< Point > point_cloud(nb_points);

  //read in data points
  double* data_ptr = data;
  for (int row=0; row < nb_points; ++row){
      // create new point and corresponding vertex
      ANNPoint p(point_dim);
      p.coord = new double[point_dim];
      for (int col=0; col<point_dim; col++){
	    p.coord[col] = *data_ptr;
	    //cout << p.coord[col] << " ";
	    ++data_ptr;
      }
      //cout << endl;
      Point v(p);
      v.data.boundary_flag=false;
      point_cloud[row] = v;
    }

  //cout << "Number of input points: " << nb_points << endl;

  //create distance structure
  Distance_ANN< vector< Point >::iterator > metric_information;
  metric_information.initialize(point_cloud.begin(),
				point_cloud.end(),
				point_dim);
  
  //compute density

  distance_to_density(point_cloud.begin(),point_cloud.end(),
		      num_neighb, metric_information);


  // sort point cloud and retrieve permutation (for pretty output)
  vector<int> perm;
  perm.reserve(nb_points);
  for(int i=0; i < nb_points; i++)
    perm.push_back(i);
  std::sort(perm.begin(), perm.end(), Less_Than<vector<Point> >(point_cloud));
  // store inverse permutation as array of iterators on initial point cloud
  vector< vector<Point>::iterator> pperm;
  pperm.reserve(nb_points);
  for (int i=0; i<nb_points; i++)
    pperm.push_back(point_cloud.begin());
  for (int i=0; i<nb_points; i++)
    pperm[perm[i]] = (point_cloud.begin() + i);
  // operate permutation on initial point cloud 
  vector<Point> pc;
  pc.reserve(nb_points);
  for (int i=0; i<nb_points; i++)
    pc.push_back(point_cloud[i]);
  for (int i=0; i<nb_points; i++)
    point_cloud[i] = pc[perm[i]];
  
  //update distance structure --- since it relies on the order of entry
  metric_information.initialize(point_cloud.begin(),point_cloud.end(), point_dim);

  //set rips parameter
  metric_information.mu = r*r;

  //create cluster data structure
  Clustering< vector<Point>::iterator > clustering;
  //set threshold
  clustering.tau = persistence_threshold;
  
  // perform clustering
  compute_persistence(point_cloud.begin(),point_cloud.end(), metric_information, clustering);

  if (clusters){
    // compress data structure:
    // attach each data point to its cluster's root directly
    // to speed up output processing
    attach_to_clusterheads(point_cloud.begin(),point_cloud.end());

    // output clusters (use permutation to preserve original point order)
    clustering.get_clusters(pperm.begin(), pperm.end(), clusters);
  }

  if (diagram){
    clustering.get_diagram(diagram);
  }
}

void diagram(double* data, int nb_points, int point_dim, int num_neighb, double r, std::vector<PersistanceDiagramPoint>* diagram){
  double persistence_threshold = std::numeric_limits<double>::infinity();
  cluster_base(data, nb_points, point_dim, num_neighb, r, persistence_threshold, NULL, diagram);
}

void cluster(double* data, int nb_points, int point_dim, int num_neighb, double r, double persistence_threshold, std::vector<Cluster>* clusters){
  cluster_base(data, nb_points, point_dim, num_neighb, r, persistence_threshold, clusters, NULL);
}
