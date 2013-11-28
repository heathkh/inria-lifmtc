
import base64
import json
import string
import StringIO
from PIL import Image

from snap.pyglog import *
from snap.google.base import py_base


def JpegToDataUrl(jpeg_data):
  dataurl = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_data)
  return dataurl


def UnityRgbToCssString(rgb):
  return 'rgb(%d,%d,%d)' % (rgb[0]*255, rgb[1]*255, rgb[2]*255)

def RenderMatchSVG(image_a_filename,
                   image_b_filename,
                   clusters_to_matches, # list of nx6 matrices where rows are matching point coords [ax ay ar bx by br]
                   colors):
    svg = None
    image_a_data = open(image_a_filename,'rb').read()
    image_b_data = open(image_b_filename,'rb').read()
        
    image_a = Image.open(StringIO.StringIO(image_a_data))
    image_b = Image.open(StringIO.StringIO(image_b_data))
    
    image_a_width, image_a_height = image_a.size
    image_b_width, image_b_height = image_b.size 
    
    # render svg
    width = image_a_width + image_b_width
    height = max(image_a_height, image_b_height)
    
    
    svg = """<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">"""
    svg += """<style> line { opacity: 0.25;} line:hover { opacity: 1.0;} </style>"""
    svg += '<svg width="%d" height="%d" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">' % (width, height)
    svg += '<image x="%dpx" y="%dpx" width="%dpx" height="%dpx" xlink:href="%s"> </image> \n' % (0, 0, image_a_width, image_a_height, JpegToDataUrl(image_a_data));    
    svg += '<image x="%dpx" y="%dpx" width="%dpx" height="%dpx" xlink:href="%s"> </image> \n' % (image_a_width, 0, image_b_width, image_b_height, JpegToDataUrl(image_b_data));
    
    
    match_info = []
    for cluster_index, matches in enumerate(clusters_to_matches):      
      color_css = UnityRgbToCssString(colors[cluster_index])
      for correspondence_index, match_points in enumerate(matches):
        a_x, a_y, a_r, b_x, b_y, b_r = match_points        
        b_x += image_a_width
        
        #svg += '<line x1="%d" y1="%d" x2="%d" y2="%d" style="stroke:%s;stroke-width:2"/>\n' % (x1,y1,x2,y2, color)
        
        left_pt_id = "c%d_lp%d" %(cluster_index, correspondence_index)
        right_pt_id = "c%d_rp%d" % (cluster_index, correspondence_index)

        svg += "<circle id=\"%s\" cx=\"%f\" cy=\"%f\" r=\"4\" stroke=\"black\" stroke-width=\"0\" fill=\"%s\"/>\n" % (left_pt_id, a_x, a_y, color_css)
        svg += "<circle id=\"%s\" cx=\"%f\" cy=\"%f\" r=\"4\" stroke=\"black\" stroke-width=\"0\" fill=\"%s\"/>\n" % (right_pt_id, b_x, b_y, color_css)
       
        svg += "<circle id=\"%s_support\" cx=\"%f\" cy=\"%f\" r=\"%f\" stroke-width=\"4\" fill=\"none\" opacity=\"0.5\" stroke=\"%s\" >\n" % (left_pt_id, a_x, a_y, a_r, color_css)
        svg += "<set attributeName=\"opacity\" from=\"0.5\" to=\"1.0\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" % (left_pt_id, left_pt_id)
        svg += "<set attributeName=\"opacity\" from=\"0.5\" to=\"1.0\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" % (right_pt_id, right_pt_id)
        svg += "</circle>"

        svg += "<circle id=\"%s_support\" cx=\"%f\" cy=\"%f\" r=\"%f\" stroke-width=\"4\" fill=\"none\" opacity=\"0.5\" stroke=\"%s\" >\n" % ( right_pt_id, b_x, b_y, b_r, color_css)
        svg += "<set attributeName=\"opacity\" from=\"0.5\" to=\"1.0\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" %(left_pt_id, left_pt_id)
        svg += "<set attributeName=\"opacity\" from=\"0.5\" to=\"1.0\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" %(right_pt_id, right_pt_id)
        svg += "</circle>"

        svg += "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" style=\"stroke:%s;stroke-width:4\" visibility=\"visible\">" % (a_x, a_y, b_x, b_y, color_css)
        svg += "<set attributeName=\"visibility\" from=\"hidden\" to=\"visible\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" % ( left_pt_id, left_pt_id)
        svg += "<set attributeName=\"visibility\" from=\"hidden\" to=\"visible\" begin=\"%s.mouseover\" end=\"%s.mouseout\"/>" % (right_pt_id, right_pt_id)
        svg += "</line>"
        
    svg += '</svg>'
    
    return svg
