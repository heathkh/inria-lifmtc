#!/usr/bin/python

import string
import time
import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import math 
import colorsys 
import cStringIO
import cPickle as pickle
from snap.iw import iw_pb2
from snap.iw.matching import py_featureextractor
from snap.iw.matching import py_featurematcher
from snap.pyglog import *
from snap.tomato import py_tomato
from mpl_toolkits.mplot3d import Axes3D

import vis
import image_matcher_config

from snap.iw.matching import py_imagematcher


def CreateRansacMatcher():
  config = image_matcher_config.GetImageMatcherConfig('usift')
  matcher = py_imagematcher.CreateImageMatcherOrDie(config)
  return matcher

def GenerateDistinctColorsRGB(N, alpha = None):
  HSV_tuples = [(x*1.0/N, 0.9, 1.0) for x in range(N)]
  RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
  RGBA_tuples = [[rgb[0], rgb[1], rgb[2], alpha] for rgb in RGB_tuples]
  return RGBA_tuples

def CreateFeatureExtractor():
  params = iw_pb2.FeatureExtractorParams()
  params.ocv_sift_params.num_octave_layers = 3
  params.ocv_sift_params.contrast_threshold = 0.04
  params.ocv_sift_params.edge_threshold = 30
  params.ocv_sift_params.sigma = 1.2
  params.ocv_sift_params.upright = True
  params.ocv_sift_params.root_sift_normalization = False
  CHECK(params.ocv_sift_params.IsInitialized())  
  extractor = py_featureextractor.CreateFeatureExtractorOrDie(params)
  return extractor

class Dataset(object):
  def __init__(self):
    self.imageid_to_filename = {}
    self.imageid_to_classid = {}
    self.classid_to_classname = {}
    self.imageids = []
    return
        
  def Load(self, dataset_path):
    # get list of directories / objects
    subdirs = [x[0] for x in os.walk(dataset_path) if x[0] != dataset_path]    
    for class_id, subdir in enumerate(subdirs):      
      files = glob.glob('%s/*.jpg' % (subdir))
      classname = os.path.basename(subdir)
      print 'classname: %s' % (classname)
      self.classid_to_classname[class_id] = classname
      for file in files:
        image_filename = os.path.basename(file)
        image_id = int(image_filename[0:-4])
        self.imageids.append(image_id)
        print ' %d' % (image_id)
        self.imageid_to_filename[image_id] = file
        self.imageid_to_classid[image_id] = class_id    
    return


def FilterTomatoCorrespondences(in_correspondences, features_a, features_b):    
  # find the smallest match distance for each point... 
  num_features_a = len(features_a.keypoints)
  num_features_b = len(features_b.keypoints)
  indexa_to_minmatchdist = [1e10]*num_features_a
  indexb_to_minmatchdist = [1e10]*num_features_b
  for c in in_correspondences:
    ia, ib = c.index_a, c.index_b
    indexa_to_minmatchdist[ia] = min(indexa_to_minmatchdist[ia], c.dist)
    indexb_to_minmatchdist[ib] = min(indexb_to_minmatchdist[ib], c.dist)
  
  out_correspondences = []    
  for c in in_correspondences:
    ia, ib = c.index_a, c.index_b                            
    # discard matches that are not close to the best match distance involving this point
    # helps remove matches to the same point that are much worse than the best 
    if c.dist > 1.1*indexa_to_minmatchdist[ia] or c.dist > 1.1*indexb_to_minmatchdist[ib]:
      continue
    out_correspondences.append(c)
  return out_correspondences

class ImagePairData(object):
  def __init__(self):
    self.features_a = None
    self.features_b = None
    return

def ExtractFeatures(dataset):  
  imagepair_data = {} # maps (image_a_id, image_b_id) to 
  imageid_to_imagefeatures = {}
  extractor = CreateFeatureExtractor()
  
  for image_id in dataset.imageids:
    filename = dataset.imageid_to_filename[image_id]
    imagedata = open(filename, 'rb').read()
    ok, features = extractor.Run(imagedata)
    CHECK(ok)
    imageid_to_imagefeatures[image_id] = features
  return imageid_to_imagefeatures


def LoadData(dataset_name):
  dataset = Dataset()  
  dataset.Load('../data/%s' % (dataset_name))
  imagepair_data = None      
  match_data_cache_filename = 'features_cache_%s.dat' % (dataset_name)
  if os.path.exists(match_data_cache_filename):
    print 'Using data from cache: %s ' % (match_data_cache_filename)
    imagepair_data = pickle.load(open(match_data_cache_filename, 'rb')) 
  else:  
    imagepair_data = ExtractFeatures(dataset)
    pickle.dump(imagepair_data, open(match_data_cache_filename, 'wb'))
  return dataset, imagepair_data


def GenerateTomatoClusters(features_a, features_b, num_neighbors, rips_radius, persistence_threshold):
  
  num_features_a = len(features_a.keypoints)
  num_features_b = len(features_b.keypoints)
  CHECK_EQ(len(features_a.keypoints), len(features_a.descriptors))
  CHECK_EQ(len(features_b.keypoints), len(features_b.descriptors))
  matcher = py_featurematcher.BidirectionalFeatureMatcher()
  ok, correspondences_a_or_b, correspondences_a_and_b = matcher.Run(features_a, features_b)
  CHECK(ok)
  correspondences_tomato = FilterTomatoCorrespondences(correspondences_a_or_b, features_a, features_b)
  tranform_points = []
  for c in correspondences_tomato:
    ia, ib = c.index_a, c.index_b        
    ka, kb = features_a.keypoints[ia], features_b.keypoints[ib]
    scale_atob = kb.radius / ka.radius
    # apply inverse of scaling a->b to put pos in b in same scale as pos in a        
    tx_atob = kb.pos.x/scale_atob - ka.pos.x
    ty_atob = kb.pos.y/scale_atob - ka.pos.y
    log_scale_atob = math.log(scale_atob,2)
    tranform_point = [tx_atob, ty_atob, log_scale_atob, ka.pos.x, ka.pos.y, ka.radius, kb.pos.x, kb.pos.y, kb.radius]        
    tranform_points.append(tranform_point)  
  
  tomato_data = np.array(tranform_points)    
  logscale_stretch = 4  
  
  transform_data = tomato_data[:,0:3] # first 3 columns are dx, dy, ds
  # rescale the scale so it has similar range as dx dy
  transform_data[:,2] = transform_data[:,2]*logscale_stretch
  tomato_clusters = py_tomato.cluster(transform_data, num_neighbors, rips_radius, persistence_threshold)
  # keep only clusters with at least 10 votes
  #min_votes = 10
  min_votes = 3
  top_clusters = [c for c in tomato_clusters if len(c.member_indices) > min_votes]
  # keep only top 3 clusters    
  top_clusters = top_clusters[0:3]
  return transform_data, tomato_data, top_clusters

def RenderFigureToSvg():
  mem_file = cStringIO.StringIO()
  plt.savefig(mem_file, format='svg')
  svg = mem_file.getvalue()
  mem_file.close()
  return svg

def RenderTomatoClusterResults(image_a_filename, image_b_filename, transform_data, tomato_data, top_clusters, tomato_diagram):
  num_points = tomato_data.shape[0]
  num_nontrival_clusters = len(top_clusters)
  colors = GenerateDistinctColorsRGB(num_nontrival_clusters, 1.0)
  index_to_color = [[0,0,0, 0.1]]*num_points
  clusters_to_matches = []
  for i, cluster in enumerate(top_clusters):
    indices = [v for v in cluster.member_indices]
    cluster_size = len(indices)      
    for index in indices:
      index_to_color[index] = colors[i]      
    print i, cluster.persistence, cluster_size      
    cluster_matches = tomato_data[indices,3:10]      
    clusters_to_matches.append(cluster_matches)
  
  match_svg = vis.RenderMatchSVG(image_a_filename, image_b_filename, clusters_to_matches, colors)
  
  cluster_info_table = ''
  if top_clusters:      
    cluster_info_table += '<table>'
    cluster_info_table += '<tr><th>Cluster</th><th>Persistence</th><th>Size</th></tr>'
    for i, cluster in enumerate(top_clusters):
      indices = [v for v in cluster.member_indices]
      cluster_size = len(indices)                    
      cluster_info_table += '<tr>'
      cluster_info_table += '<tr><td style="background-color:%s">%d</td><td>%f</td><td>%d</td></tr>' % (vis.UnityRgbToCssString(colors[i]), i, cluster.persistence, cluster_size )
      cluster_info_table += '</tr>'      
    cluster_info_table += '</table>'

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')        
  ax.scatter(transform_data[:,0], transform_data[:,1], transform_data[:,2], c=index_to_color, marker='o')
  ax.set_xlim3d(-200,200)
  ax.set_ylim3d(-200,200)    
  ax.set_ylim3d(-200,200)
  ax.set_xlabel('DX')
  ax.set_ylabel('DY')
  ax.set_zlabel('DS')    
  transform_space_svg = RenderFigureToSvg()
  
  # draw persistence diagram
  fig = plt.figure()
  x = []
  y = []  
  for p in tomato_diagram:
    x.append(p.birth)
    y.append(p.death)
    
  lim = max(max(x), max(y))*1.1  
  plt.plot(x,y,'ro', [0, lim], [0, lim], 'k--')
  plt.xlabel('Birth')
  plt.ylabel('Death')
  plt.title('Persistence Diagram')
  
  
  plt.xlim([0, lim])
  plt.ylim([0, lim])
  #plt.show()
  
  persistence_diagram_svg = RenderFigureToSvg()  
  return cluster_info_table, transform_space_svg,  match_svg, persistence_diagram_svg


def RenderRansacResults(image_a_filename, image_b_filename, ransac_geometric_matches):
  
  num_ransac_matches = len(ransac_geometric_matches.entries)
  
  colors = GenerateDistinctColorsRGB(num_ransac_matches, 1.0)
  
  match_sets = []
  for gm in ransac_geometric_matches.entries:
    matches = []
    for c in gm.correspondences:
      matches.append([c.a.pos.x, c.a.pos.y, c.a.radius, c.b.pos.x, c.b.pos.y, c.b.radius])
    match_sets.append(matches)  
  
  match_svg = vis.RenderMatchSVG(image_a_filename, image_b_filename, match_sets, colors)
  return match_svg


def RenderParamsHtml(rips_radius, num_neighbors, persistence_threshold, time_tomato, time_ransac):
  html = ''
  html += '<ul>'
  html += '<li> rips_radius: %f' % (rips_radius)
  html += '<li> num_neighbors: %d' % (num_neighbors)
  html += '<li> persistence_threshold: %f' % (persistence_threshold)
  html += '<li> time_tomato: %f' % (time_tomato)
  html += '<li> time_ransac: %f' % (time_ransac)
  html += '</ul>'
  return html

class ErrorRateExperiment(object):
  def __init__(self):
    self.tp = 0
    self.fp = 0
    self.fn = 0
    self.tn = 0
    self.total_time = 0
    return
  
  def AvgTime(self):
    return self.total_time / (self.tp + self.fp + self.fn + self.tn)
    
  def AddResult(self, actual, predicted, time):
    CHECK(actual in [True, False])
    CHECK(predicted in [True, False])
    if actual:
      if predicted:
        self.tp += 1.0
      else:
        self.fn += 1.0
    else:      
      if predicted:
        self.fp += 1.0
      else:
        self.tn += 1.0  
    self.total_time += time    
    return        
  
  def Precision(self):
    return self.tp / (self.tp+self.fp)
  
  def Recall(self):
    return self.tp / (self.tp+self.fn)
  
  def FScore(self, beta = 1.0):
    precision = self.Precision()
    recall = self.Recall()
    eps = 1e-16
    f = (1.0+beta*beta)*precision*recall/(beta*beta*precision + recall + eps)
    return f

  def __str__(self):
    return 'P: %f R: %f F: %f  t:%f' % (self.Precision(), self.Recall(), self.FScore(), self.AvgTime())          

def main():  
  dataset_name = 'set04'
  
  output_results_html = False
  
  rips_radius=9
  num_neighbors=20  
  persistence_threshold=0.0015

#   rips_radius=20
#   num_neighbors=20
#   persistence_threshold=0.002

#   rips_radius=30
#   num_neighbors=20
#   persistence_threshold=0.002
      
  dataset, imageid_to_imagefeatures = LoadData(dataset_name)
  ransac = CreateRansacMatcher()
  num_images = len(dataset.imageids)
  
  t_exp =  ErrorRateExperiment()
  r_exp =  ErrorRateExperiment()
  tr_exp =  ErrorRateExperiment()
  
  
  
  template_filename = 'exp_result_template.html'
  html_template = string.Template(open(template_filename, 'r').read())
  
  
  for i in range(num_images):
    image_a = dataset.imageids[i]
    for j in range(i+1, num_images):  
      
      image_b = dataset.imageids[j]
      CHECK_NE(image_a, image_b)
      features_a = imageid_to_imagefeatures[image_a]
      features_b = imageid_to_imagefeatures[image_b]
              
      classid_a, classid_b = [dataset.imageid_to_classid[id] for id in [image_a, image_b]]
      image_a_filename, image_b_filename = [dataset.imageid_to_filename[id] for id in [image_a, image_b]]
      are_same_class = (classid_a == classid_b)    
      print '%s \n%s' % (image_a_filename, image_b_filename)
            
      time_start = time.time()                    
      transform_data, tomato_data, top_clusters = GenerateTomatoClusters(features_a, features_b, num_neighbors, rips_radius, persistence_threshold)
      time_tomato = time.time() - time_start
      tomato_found_match = (len(top_clusters) > 0)
      
      time_start = time.time()
      ransac_found_match, ransac_results, cpptime = ransac.Run(features_a, features_b)
      time_ransac = time.time() - time_start
      
      print 'tomato: %s ransac: %s' % (tomato_found_match, ransac_found_match)
        
      if not are_same_class and ransac_found_match:
        print 'Look at me! ****************************'
      
      
      
      if output_results_html:
        tomato_diagram = py_tomato.diagram(transform_data, num_neighbors, rips_radius)
        cluster_info_table, transform_space_svg,  cluster_match_svg, persistence_diagram_svg = RenderTomatoClusterResults(image_a_filename, image_b_filename, transform_data, tomato_data, top_clusters, tomato_diagram)            
        ransac_match_svg = RenderRansacResults(image_a_filename, image_b_filename, ransac_results)
        params_html = RenderParamsHtml(rips_radius, num_neighbors, persistence_threshold, time_tomato, time_ransac)        
        
        html = html_template.substitute({'cluster_info_table' : cluster_info_table,                                         
                                         'transform_space_svg' : transform_space_svg,
                                         'cluster_match_svg' : cluster_match_svg,
                                         'persistence_diagram_svg': persistence_diagram_svg,
                                         'ransac_match_svg' : ransac_match_svg,
                                         'params_html': params_html })
        prefix = 'sameclass' if are_same_class else 'diffclass'     
        result_filename = 'results/%s_class%d_class%d_image%s_to_image%s.html' % (prefix, classid_a, classid_b, image_a, image_b)
        open(result_filename, 'w').write(html)
        
      
      # evaluate error rates            
      t_exp.AddResult(are_same_class, tomato_found_match, time_tomato)
      r_exp.AddResult(are_same_class, ransac_found_match, time_ransac)
            
      time_tr = time_tomato
      tr_found_match = False
      if tomato_found_match:
        time_tr += time_ransac
        if ransac_found_match:
          tr_found_match = True      
      tr_exp.AddResult(are_same_class, tr_found_match, time_tr)
      
  
  print 't  : %s' % (t_exp)
  print 'r  : %s' % (r_exp)
  print 't+r: %s' % (tr_exp)
  
  
    
  return

if __name__ == "__main__":
  main()