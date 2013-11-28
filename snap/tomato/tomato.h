#include "Cluster_Advanced.h"


void diagram(double* data, int nb_points, int point_dim, int num_neighb, double r, std::vector<PersistanceDiagramPoint>* diagram);

void cluster(double* data, int nb_points, int point_dim, int num_neighb, double r, double persistence_threshold, std::vector<Cluster>* clusters);
