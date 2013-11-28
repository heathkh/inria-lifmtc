#!/usr/bin/python

""" This script installs a customized version of protobuf 2.4.1.  
Our application requires the use of -fPIC and the CPP implementation for python
"""

import os
import urllib
import urllib2
import tarfile
import subprocess

def ExecuteCmd(cmd, quiet=False):
  result = None
  if quiet:    
    with open(os.devnull, "w") as fnull:    
      result = subprocess.call(cmd, shell=True, stdout = fnull, stderr = fnull)
  else:
    result = subprocess.call(cmd, shell=True)
  return result

def EnsurePath(path):
  try:
    os.makedirs(path)
  except:
    pass
  return

def InstallProtobuffers():
  
  #version = '2.4.1' 
  version = '2.5.0' 
  
  url = 'http://protobuf.googlecode.com/files/protobuf-%s.tar.gz' % (version)
  
  split = urllib2.urlparse.urlsplit(url)
  dest_filename = "/tmp/" + split.path.split("/")[-1]
  urllib.urlretrieve(url, dest_filename)
  assert(os.path.exists(dest_filename))
  tar = tarfile.open(dest_filename)
  tar.extractall('/tmp/')
  tar.close()  
  src_path = '/tmp/protobuf-%s/' % (version)
  assert(os.path.exists(src_path))  
  cmd = 'cd %s && export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp && export CCFLAGS=-fPIC && export CXXFLAGS=-fPIC && ./configure && make -j10 && sudo make install && cd python && sudo  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp python setup.py build &&  sudo  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp python setup.py install && sudo ldconfig' % (src_path)
  ExecuteCmd(cmd)  
  return 0


if __name__ == "__main__":
  InstallProtobuffers()  