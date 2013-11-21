#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from tomato import py_tomato

def LoadData(filename):  
  file = open(filename, 'r')
  num_points = 0
  point_dim = None
  for line in file:
    tokens = line.split()
    if point_dim:
      assert(len(tokens) == point_dim)
    else:
      point_dim = len(tokens)
    num_points += 1    
  assert(num_points > 0)
  assert(point_dim == 2)
  
  file.seek(0)
  data = np.empty( (num_points, point_dim) )  
  for row, line in enumerate(file):
    tokens = line.split()
    for col, token in enumerate(tokens):
      data[row,col] = float(token)
  return data


def main():  
  test_filename = 'toy_example_w_o_density.txt'
  data = LoadData(test_filename)
      
  #plt.plot(data[:,0], data[:,1], 'ro')
  #plt.show()    
  
  num_neighbors=200
  rips_radius=0.5
  persistence_threshold=1.5
  clusters = py_tomato.cluster(data, num_neighbors, rips_radius, persistence_threshold)
  
  for cluster in clusters:
    indices = [v for v in cluster.member_indices]
    print cluster
    plt.plot(data[indices,0], data[indices,1], 'o')
    
  plt.show()    
  
  return

if __name__ == "__main__":
  main()