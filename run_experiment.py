#!/usr/bin/python

import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import math 
import colorsys 
import cStringIO

from snap.iw import iw_pb2
from snap.iw.matching import py_featureextractor
from snap.iw.matching import py_featurematcher
from snap.pyglog import *
from snap.tomato import py_tomato

import vis

from mpl_toolkits.mplot3d import Axes3D
import cPickle as pickle


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
    

def GenerateMatchData(dataset):  
  min_radius = 2
  match_data = {} # maps (image_a_id, image_b_id) to numpy array Nx3 of similarity transform data
  imageid_to_imagefeatures = {}
  extractor = CreateFeatureExtractor()
  
  for image_id in dataset.imageids:
    filename = dataset.imageid_to_filename[image_id]
    imagedata = open(filename, 'rb').read()
    ok, features = extractor.Run(imagedata)
    CHECK(ok)
    imageid_to_imagefeatures[image_id] = features
    
  matcher = py_featurematcher.BidirectionalFeatureMatcher()
  num_images = len(dataset.imageids)  
  for i in range(num_images):
    image_a = dataset.imageids[i]
    for j in range(i+1, num_images):      
      image_b = dataset.imageids[j]
      CHECK_NE(image_a, image_b)
      
      features_a = imageid_to_imagefeatures[image_a]
      features_b = imageid_to_imagefeatures[image_b]
      CHECK_EQ(len(features_a.keypoints), len(features_a.descriptors))
      CHECK_EQ(len(features_b.keypoints), len(features_b.descriptors))
      
      ok, correspondences = matcher.Run(features_a, features_b)
      CHECK(ok)
      
      
      num_features_a = len(features_a.keypoints)
      num_features_b = len(features_b.keypoints)
      
      
      # collect data to enforce 1-1 constraint by keeping only the best match using a point
      indexa_to_minmatchdist = [1e10]*num_features_a
      indexb_to_minmatchdist = [1e10]*num_features_b
      
      indexa_to_count = [0]*num_features_a
      indexb_to_count = [0]*num_features_b
      
      correspondences_multiplicity = {}
      
      for c in correspondences:
        ia, ib = c.index_a, c.index_b
        indexa_to_minmatchdist[ia] = min(indexa_to_minmatchdist[ia], c.dist)
        indexb_to_minmatchdist[ib] = min(indexb_to_minmatchdist[ib], c.dist)
        
        indexa_to_count[ia] += 1
        indexb_to_count[ib] += 1
        
        key = (ia,ib)
        if key not in correspondences_multiplicity:
          correspondences_multiplicity[key] = 0
        correspondences_multiplicity[key] += 1  
        
        
      tranform_points = []
      for c in correspondences:
        ia, ib = c.index_a, c.index_b        
        CHECK_LT(ia, len(features_a.keypoints))
        CHECK_LT(ib, len(features_b.keypoints))                 
        ka = features_a.keypoints[ia]
        kb = features_b.keypoints[ib]
         
         
        if c.dist > 1.1*indexa_to_minmatchdist[ia] or c.dist > 1.1*indexb_to_minmatchdist[ib]:
          continue
         
        #if indexa_to_count[ia] > 4 or indexb_to_count[ib] > 4:
        #  continue
        
        # This tweak was added after observing small features are often responsible for bad matches
        # Often locking on to textures or other small repeated features
        if kb.radius < min_radius or ka.radius < min_radius:
          continue 
        
        scale_atob = kb.radius / ka.radius
        
        # apply inverse of scaling a->b to put pos in b in same scale as pos in a        
        tx_atob = kb.pos.x/scale_atob - ka.pos.x
        ty_atob = kb.pos.y/scale_atob - ka.pos.y
        
        log_scale_atob = math.log(scale_atob,2)
        
        tranform_point = [tx_atob, ty_atob, log_scale_atob, ka.pos.x, ka.pos.y, ka.radius, kb.pos.x, kb.pos.y, kb.radius]        
        tranform_points.append(tranform_point)
        
       
    
      data = np.array(tranform_points)
      key = (image_a, image_b)
      match_data[key] = data
      
  return match_data

def main():  

  dataset_name = 'set02'
  
  dataset = Dataset()  
  dataset.Load('../data/%s' % (dataset_name))
      
  match_data = None      
  match_data_cache_filename = 'matchdata_cache_%s.dat' % (dataset_name)
  
  if os.path.exists(match_data_cache_filename):
    print 'Using data from cache: %s ' % (match_data_cache_filename)
    match_data = pickle.load(open(match_data_cache_filename, 'rb')) 
  else:  
    match_data = GenerateMatchData(dataset)
    pickle.dump(match_data, open(match_data_cache_filename, 'wb'))    
  
  for (image_a, image_b), match_data in match_data.iteritems():
    transform_data = match_data[:,0:3] # first 3 columns are dx, dy, ds
    
    # rescale the scale so it has similar range as dx dy
    transform_data[:,2] = transform_data[:,2]*4
    
    classid_a = dataset.imageid_to_classid[image_a]
    classid_b = dataset.imageid_to_classid[image_b]    
    are_same_class = (classid_a == classid_b)
    
    print dataset.imageid_to_filename[image_a]
    print dataset.imageid_to_filename[image_b]    
    print 'are same class: %s' % (are_same_class)
    
    num_points = match_data.shape[0]
    
    num_neighbors=20
    rips_radius=9
    persistence_threshold=0.001
    
    clusters = py_tomato.cluster(transform_data, num_neighbors, rips_radius, persistence_threshold)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    top_clusters = [c for c in clusters if len(c.member_indices) > 10]
    
    top_clusters = top_clusters[0:3]
    
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
      cluster_matches = match_data[indices,3:10]      
      clusters_to_matches.append(cluster_matches)
    
    
    match_svg = vis.RenderMatchSVG(dataset.imageid_to_filename[image_a],
                       dataset.imageid_to_filename[image_b],
                       clusters_to_matches,
                       colors)
    
    css_style = """
    <style rel="stylesheet">
      * { margin: 0px; padding: 0px; }
          td, th {
        width: 4rem;
        height: 2rem;
        border: 1px solid #ccc;
        text-align: center;
      }
      th {
        background: lightblue;
        border-color: white;
      }
      body {
        padding: 1rem;
      }
      </style>
    """
    
    cluster_info_table = ''
    if top_clusters:      
      cluster_info_table += '<table>'
      cluster_info_table += '<tr><td>Cluster</td><td>Persistence</td><td>Size</td></tr>'
      for i, cluster in enumerate(top_clusters):
        indices = [v for v in cluster.member_indices]
        cluster_size = len(indices)      
                
        cluster_info_table += '<tr>'
        cluster_info_table += '<tr><td style="background-color:%s">%d</td><td>%f</td><td>%d</td></tr>' % (vis.UnityRgbToCssString(colors[i]), i, cluster.persistence, cluster_size )
        cluster_info_table += '</tr>'  
      
      cluster_info_table += '</table>'
          
    ax.scatter(transform_data[:,0], transform_data[:,1], transform_data[:,2], c=index_to_color, marker='o')
    ax.set_xlim3d(-200,200)
    ax.set_ylim3d(-200,200)    
    ax.set_ylim3d(-200,200)
    ax.set_xlabel('DX')
    ax.set_ylabel('DY')
    ax.set_zlabel('DS')
    
    transform_space_svg_file = cStringIO.StringIO()
    plt.savefig(transform_space_svg_file, format='svg')
    transform_space_svg = transform_space_svg_file.getvalue()
    transform_space_svg_file.close()
    
    html = css_style+ cluster_info_table + transform_space_svg  + match_svg
    
    prefix = ''
    if are_same_class:
      prefix = 'sameclass'
    else:
      prefix = 'diffclass'
    
    result_filename = 'results/%s_class%d_class%d_image%s_to_image%s.html' % (prefix, classid_a, classid_b, image_a, image_b)
    
    open(result_filename, 'w').write(html)
    
    #plt.show()
     
    
  
  
  return

if __name__ == "__main__":
  main()