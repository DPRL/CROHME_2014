##    DPRL CROHME 2014
##    Copyright (c) 2013-2014 Lei Hu, Kenny Davila, Francisco Alvaro, Richard Zanibbi
##
##    This file is part of DPRL CROHME 2014.
##
##    DPRL CROHME 2014 is free software: 
##    you can redistribute it and/or modify it under the terms of the GNU 
##    General Public License as published by the Free Software Foundation, 
##    either version 3 of the License, or (at your option) any later version.
##
##    DPRL CROHME 2014 is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with DPRL CROHME 2014.  
##    If not, see <http://www.gnu.org/licenses/>.
##
##    Contact:
##        - Lei Hu: lei.hu@rit.edu
##        - Kenny Davila: kxd7282@rit.edu
##        - Francisco Alvaro: falvaro@dsic.upv.es
##        - Richard Zanibbi: rlaz@cs.rit.edu 

from __future__ import division
import xml.dom.minidom as minidom
import math
import sys
import os
import os.path
import itertools
import collections
import codecs
from copy import deepcopy
import pylab

import requests
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
##import classifier
import symbol_classifier
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import subprocess
import shutil

sys.setrecursionlimit(10000)
## load the classifier
KENNY_CLASSIFIER = symbol_classifier.SymbolClassifier("./crohme2013_train_nojunk.rsvm", "./crohme2013_train_nojunk_params.dat")

WIDE_THRESHOLD = 2.5
NARROW_THRESHOLD = 0.3

def zipwith(fun, *args):
        return map(lambda x: fun(*x), zip(*args))
        
def distance(point1, point2):
        box = zipwith(lambda x,y: abs(x - y), point1, point2)
        return math.sqrt(sum(map(lambda x: x**2, box)))

class Equation(object):
        def __init__(self):
                super(Equation, self).__init__()
                self.strokes = {}
                self.segments_truth = SegmentSet()
                self.segments = SegmentSet()
                self.id_map = {}
                self.reverse_id_map = {}
                
        @classmethod
        def from_inkml(cls, filename):
                dom = minidom.parse(filename)
                self = cls()
                self.dom = dom

                id_flag = 0

                for node in self.dom.getElementsByTagName('trace'):
                        trace_id = int(node.getAttribute('id'))
                        points_string = node.firstChild.data.split(',')
                        points = []
                        for p in points_string:
                                points.append(tuple(map(float, p.split())))
                        ## for those (x,y,time)
                        if len(points[0])>2:
                                new_points = []
                                for i in range(len(points)):
                                        one_point = points[i]
                                        one_point_array = np.array(one_point)
                                        one_point_list = one_point_array.tolist()
                                        one_point_tuple = tuple(one_point_list[:2])
                                        new_points.append(one_point_tuple)
                                points = new_points
##                        self.strokes[trace_id] = Stroke(trace_id, points)
                        ## for the expression data, which has the stroke index not beginning from 0
                        ## debug for CROHME 2014
                        self.strokes[id_flag] = Stroke(id_flag, points)
                        self.id_map[id_flag] = trace_id
                        self.reverse_id_map[trace_id] = id_flag 
                        id_flag+=1

                for node in self.dom.getElementsByTagName('traceGroup'):
                        group = []
                        symbol = node.getElementsByTagName('annotation')[0].firstChild.data.strip()
                        if symbol == "Segmentation" or symbol == "From ITF":
                                continue
                        for stroke in node.getElementsByTagName('traceView'):
##                                group.append(int(stroke.getAttribute('traceDataRef')))
                                ## for the expression data, which has the stroke index not beginning from 0
                                ## debug for CROHME 2014
                                group.append(self.reverse_id_map[int(stroke.getAttribute('traceDataRef'))])

                        self.segments_truth.add(Segment(group, symbol))
                self.segments = SegmentSet.init_unconnected_strokes([s.id for s in self.strokes.values()])

                return self
                
        def output_inkml(self, filename):
                math = self.dom.getElementsByTagName('math')[0]
                for c in list(math.childNodes):
                        math.removeChild(c)
                        c.unlink()
                mrow = self.dom.createElement('mrow')
                math.appendChild(mrow)
                group = self.dom.getElementsByTagName('traceGroup')[0]
                for c in list(group.childNodes):
                        if c.nodeName == 'annotation':
                                continue
                        group.removeChild(c)
                        c.unlink()
                for i, seg in enumerate(sorted(self.segments, key=lambda x: min(x.strokes))):
                        mi = self.dom.createElement('mi')
                        mi.setAttribute('xml:id', 'x_%d' % i)
                        mi.appendChild(self.dom.createTextNode('x'))
                        mrow.appendChild(mi)
                        
                        tgid = int(group.getAttribute('xml:id')) + 1
                        tg = self.dom.createElement('traceGroup')
                        tg.setAttribute('xml:id', str(tgid + i))
                        an = self.dom.createElement('annotation')
                        an.setAttribute('type', 'truth')
                        an.appendChild(self.dom.createTextNode('x'))
                        tg.appendChild(an)
                        for stroke in seg.strokes:
                                tv = self.dom.createElement('traceView')
                                tv.setAttribute('traceDataRef', str(stroke))
                                tg.appendChild(tv)
                        ml = self.dom.createElement('annotationXML')
                        ml.setAttribute('href', 'x_%d' % i)
                        tg.appendChild(ml)
                        group.appendChild(tg)
                        
                        
                with codecs.open(filename,'w','utf-8') as f:
                        self.dom.writexml(f, encoding='utf-8')

        def contains_true_segmentation(self):
                return self.segments_truth in self.segments.combinations()

        def test_fuzzy_segments(self):
                all_segs = set()
                for fs in self.segments:
                        c = fs.combinations()
                        for s in c:
                                all_segs.update(s)

                correct = len(all_segs.intersection(self.segments_truth))
                return (correct, len(all_segs), len(self.segments_truth))
                
        def test_segments(self):
                correct = len(self.segments.intersection(self.segments_truth))                  
                return (correct, len(self.segments), len(self.segments_truth))
                
        def test_segments_by_symbol(self):
                ret = {}
                for s in self.segments_truth:
                        correct = 1 if s in self.segments else 0
                        if s.symbol in ret:
                                ret[s.symbol] = tuple(zipwith(lambda x,y: x + y, ret[s.symbol], (correct, 1)))
                        else:
                                ret[s.symbol] = (correct, 1)
                return ret
                
        def segment_for_stroke(self, i):
                if not hasattr(self, '_segments_by_strokes'):
                        self._segments_by_strokes = {}
                        for s in self.segments:
                                for st in s.strokes:
                                        self._segments_by_strokes[st] = s
                return self._segments_by_strokes[i]
                
        def segment_truth_for_stroke(self, i):
                if not hasattr(self, '_segments_truth_by_strokes'):
                        self._segments_truth_by_strokes = {}
                        for s in self.segments_truth:
                                for st in s.strokes:
                                        self._segments_truth_by_strokes[st] = s
                try:
                        return self._segments_truth_by_strokes[i]
                except:
                        return Segment(['x'])
                        
        def missed_symbol(self, symbol):
                for s in self.segments_truth:
                        if s.symbol == symbol and s not in self.segments:
                                return True
                return False
        
        def test_classification(self):
                correct = 0
                for st in self.strokes.keys():
                        if self.segment_for_stroke(st).symbol == self.segment_truth_for_stroke(st).symbol:
                                correct += 1
                return (correct, len(self.strokes))
                

        def lei_CROHME2013_segment(self):
                 
                  O_eq = copy.deepcopy(self)
                  ## merge touching strokes
                  self.merge_touching()

                  ## preprocessing the input equation
                  self = equation_preprocessing(self)
                  pairs = zip(self.strokes.values(), self.strokes.values()[1:])

                  for s1, s2 in pairs:
                          current_stroke = s1
                          next_stroke = s2
                          eq = self

                          ## foreground shape context feature
                          foreground_scf = current_stroke.context_shape_features_1NN(next_stroke)
                          ## background shape context feature
                          background_scf = get_3NN_background_scf(eq, current_stroke)
                          ## global shape context feature
                          global_scf = get_global_scf(eq,current_stroke)

                          temp_feature = current_stroke.features(next_stroke)

                          O_current_stroke = O_eq.strokes[s1.id]
                          O_next_stroke = O_eq.strokes[s2.id]
                          ## get the two sets of the classification scores
                          two_CC = get_two_CC(O_current_stroke, O_next_stroke)

                          ## get all the features
                          all_feature = foreground_scf + background_scf + global_scf + temp_feature + two_CC
                          all_feature = np.array(all_feature)

                          ## get the PCA coefficient
                          COEFF = np.loadtxt('TrainCOEFF.txt', delimiter = ',')
                          ## load the AdaBoost classifier
                          all_classifiers = np.loadtxt('Train5000iteration', delimiter = ',')

                          ## get the PCA features
                          PCA_all_feature = []
                          component_num = 100
                          for i in range(component_num):
                                  one_component = COEFF[:,i]
                                  PCA_all_feature.append(sum(all_feature*one_component))
                                  
                          ## use the AdaBoost classifier to do the classification
                          sum_h = 0
                          one_feature_vector = np.array(PCA_all_feature)
                          iteration_num = len(all_classifiers)
                       
                          for i in range(iteration_num):
                                  one_feature = one_feature_vector[int(all_classifiers[i][0])]
                                  if (one_feature < all_classifiers[i][1]):
                                          sum_h += (1.0*all_classifiers[i][2])
                                  else:
                                          sum_h += (1.0*all_classifiers[i][3])

                          if sum_h > 0:
                                  self.segments.merge_strokes(s1.id, s2.id)

                                                
        def merge_touching(self):
                for s1, s2 in itertools.combinations(self.strokes.values(), 2):
                        if s1.bb_intersects(s2):
                                if s1.intersects(s2):
                                        self.segments.merge_strokes(s1.id, s2.id)
                        
        def find_closest_stroke(self, stroke):
                d = 9001
                closest = -1
                for id, s in self.strokes.items():
                        if s != stroke:
                                if d > closest_distance(s, stroke):
                                        closest = id
                                        d = closest_distance(s, stroke)
                return closest
                
        def avg_extents(self):
                if not hasattr(self, '_avg_extents'):
                        widths = []
                        heights = []
                        diags = []
                        for s in self.strokes.values():
                                mins, maxs = s.extents
                                widths.append(maxs[0] - mins[0])
                                heights.append(maxs[1] - mins[1])
                                diags.append(s.half_diag)       
                        avg_width = median(widths)
                        avg_height = median(heights)
                        avg_diag = median(diags)
                        self._avg_extents = avg_width, avg_height, avg_diag
                return self._avg_extents
                
        def get_wide_strokes(self):
                avg_width = self.avg_extents()[0]
                wide_strokes = set()
                for s in self.strokes.values():
                        if s.width > WIDE_THRESHOLD * avg_width:
                                wide_strokes.add(s)
                return wide_strokes
                
        def get_dots(self, thresh):
                avg_width, avg_height, avg_diag = self.avg_extents()
                dots = set()
                for s in self.strokes.values():
                        if s.half_diag < thresh * avg_diag:
                                dots.add(s)
                return dots
                
        def merge_dots(self):
                avg_width, avg_heigh, avg_diag  = self.avg_extents()
                for s in self.get_dots(NARROW_THRESHOLD):
                        neighbors = []
                        if s.id - 1 in self.strokes:
                                neighbors.append(self.strokes[s.id - 1])
                        if s.id + 1 in self.strokes:
                                neighbors.append(self.strokes[s.id + 1])
                        closest = reduce(lambda x,y: x if x.closest_distance(s) < y.closest_distance(s) else y, neighbors)
                        self.segments.merge_strokes(s.id, closest.id)

                                
def median(l):
        ls = sorted(l)
        n = len(l)
        if n % 2 == 1:
                return ls[int((n - 1) / 2)]
        else:
                low = ls[int((n / 2) - 1)]
                high = ls[int(n / 2)]
                return float(low + high) / 2
                

class SegmentSet(set):
        def __init__(self, *args, **kwargs):
                super(SegmentSet, self).__init__(*args, **kwargs)
                self.prob = 1.0

        @classmethod
        def init_unconnected_strokes(cls, strokes):
                return cls([Segment([s]) for s in strokes])

        def __repr__(self):
                segs = []
                for s in self:
                        segs.append(sorted(s.strokes))
                segs.sort(key=lambda x:x[0])
                return '%s : %.6f' % ('; '.join([', '.join(map(str, s)) for s in segs]), self.prob)

        def biggest_segment(self):
                return max([len(s.strokes) for s in self])

        def merge_strokes(self, first, second):
                seg1 = None
                seg2 = None
                for s in self:
                        if first in s:
                                seg1 = s
                        if second in s:
                                seg2 = s
                if seg1 != seg2:
                        self.remove(seg1)
                        self.remove(seg2)
                        self.add(seg1.union(seg2))

class FuzzySegmentSet(SegmentSet):
        @classmethod
        def init_unconnected_strokes(cls, strokes):
                return cls([FuzzySegment([s]) for s in strokes])

        @classmethod
        def from_segment_set(cls, sset):
                return cls([FuzzySegment(s.strokes) for s in sset])

        def merge_strokes(self, first, second, prob=1.0):
                seg1 = None
                seg2 = None
                for s in self:
                        if first in s:
                                seg1 = s
                        if second in s:
                                seg2 = s
                if seg1 != seg2:
                        self.remove(seg1)
                        self.remove(seg2)
                        self.add(seg1.union(seg2, {(first, second): prob}))

        def combinations(self):
                sets = [SegmentSet()]
                for fs in self:
                        newsets = []
                        for s in sets:
                                for c in fs.combinations():
                                        new = deepcopy(s)
                                        new.update(c)
                                        new.prob *= c.prob
                                        newsets.append(new)
                        sets = newsets
                return sets

        def best_combination(self):
                s = SegmentSet()
                for fs in self:
                        best = fs.best_combination()
                        s.update(best)
                        s.prob *= best.prob
                return s

        def num_combs(self):
                p = 1
                for fs in self:
                        p *= len(fs.combinations())
                return p

        def limit_size(self):
                while True:
                        biggest = list(self)[0]
                        for s in self:
                                if len(s.strokes) > len(biggest.strokes):
                                        biggest = s
                        if len(biggest.strokes) <= 10:
                                break
                        newsegs = biggest.split_weakest()
                        self.remove(biggest)
                        self.update(newsegs)
                        
                                

class Segment(object):
        def __init__(self, strokes, symbol='x'):
                self.strokes = frozenset(strokes)
                self.symbol = symbol
                
        def __hash__(self):
                return self.strokes.__hash__()
                
        def __eq__(self, other):
                return self.strokes == other.strokes
                
        def __ne__(self, other):
                return not self == other
                
        def __contains__(self, item):
                return item in self.strokes
                
        def __repr__(self):
                return 'Segment(%s, \'%s\')' % (repr(self.strokes), self.symbol)
                
        def union(self, other):
                return Segment(self.strokes.union(other.strokes), self.symbol)
                
        def intersection(self, other):
                return Segment(self.strokes.intersection(other.strokes), self.symbol)

class FuzzySegment(Segment):
        def __init__(self, strokes, transitions={}, symbol='x'):
                super(FuzzySegment, self).__init__(strokes, symbol)
                self.transitions = transitions

        def union(self, other, newtransition={}):
                strokes = self.strokes.union(other.strokes)
                transitions = dict(self.transitions.items() + other.transitions.items() + newtransition.items())
                return FuzzySegment(strokes, transitions)

        def combinations(self, max_group=4):
                segments = [SegmentSet.init_unconnected_strokes(self.strokes)]
                for (src, dst), p in sorted(self.transitions.items(), key=lambda x: x[1], reverse=True):
                        together = []
                        if p > 0.0:
                                for s in segments:
                                        s2 = deepcopy(s)
                                        s2.merge_strokes(src, dst)
                                        s2.prob *= p
                                        if s2.biggest_segment() <= max_group:
                                                together.append(s2)
                        if p < 1.0:
                                for s in segments:
                                        s.prob *= (1.0 - p)
                        else:
                                segments = []
                        segments.extend(together)
                return segments

        def best_combination(self):
                s = SegmentSet.init_unconnected_strokes(self.strokes)
                for (src, dst), p in sorted(self.transitions.items(), key=lambda x: x[1], reverse=True):
                        if p > 0.5:
                                s.merge_strokes(src, dst)
                                s.prob *= p
                        else:
                                s.prob *= (1.0 - p)
                return s

        def split_weakest(self):
                newset = FuzzySegmentSet.init_unconnected_strokes(self.strokes)
                transitions = sorted(self.transitions.items(), key=lambda x: x[1], reverse=True)[:-1]
                for (src, dst), p in transitions:
                        newset.merge_strokes(src, dst, p)
                return newset
                
class Stroke(object):
        def __init__(self, id, points):
                self.id = id
                self.points = points
        
        def __eq__(self, other):
                return self.id == other.id and self.points == other.points
                
        def __ne__(self, other):
                return not self == other

        def __hash__(self):
                return hash((id, str(self.points)))

        def __repr__(self):
                return 'Stroke(id=%d)' % self.id
                
        @property
        def extents(self):
                if not (hasattr(self, '_mins') and hasattr(self, '_maxs')):
                        mins = list(self.points[0])
                        maxs = list(self.points[0])
                
                        for point in self.points:
                                mins = zipwith(min, mins, point)
                                maxs = zipwith(max, maxs, point)
                        self._mins, self._maxs = tuple(mins), tuple(maxs)
                return self._mins, self._maxs
                
        @property
        def center(self):
                mins, maxs = self.extents
                box = zipwith(lambda x,y: abs(x - y), mins, maxs)
                return zipwith(lambda m,b: m + (float(b) / 2), mins, box)
                
        @property
        def half_diag(self):
                return distance(*self.extents) / 2
                
        @property
        def width(self):
                mins, maxs = self.extents
                return maxs[0] - mins[0]
                
        @property
        def height(self):
                mins, maxs = self.extents
                return maxs[1] - mins[1]

        ## define the area
        @property
        def area(self):
                return (self.width * self.height)
                
        def average_diag(self, other):
                avg_diag = (self.half_diag + other.half_diag) / 2
                if avg_diag == 0:
                        avg_diag = 0.01
                return avg_diag


        def center_distance(self, other):
                return distance(self.center, other.center)


        def closest_distance(self, other):
                ret = distance(self.points[0], other.points[0])
                for x in self.points:
                        for y in other.points:
                                ret = min(ret, distance(x, y))
                return ret

        ## define the farest distance
        def farest_distance(self, other):
                ret = distance(self.points[0], other.points[0])
                for x in self.points:
                        for y in other.points:
                                ret = max(ret, distance(x, y))
                return ret
        

        ## get all the features
        
        def features(self, other):
                 all_features = []                 
                 ## 1st feature, minimal distance between the two strokes/average of diagonal
                 if (self.average_diag(other) * 2):
                         first_feature = self.closest_distance(other) / (self.average_diag(other) * 2)
                 else:
                         first_feature = 0
                 all_features.append(first_feature)

                 ## second, third and forth features are from Shi's paper "a unified framework for symbol segmentation and
                 ## recognition of handwritten mathematical expressions". But the details how to get the horizontal, vertical and size thresholds are missing
                 Points1 = self.points
                 Points2 = other.points
                 PointsArray1 = np.array(Points1)
                 PointsArray2 = np.array(Points2)
                 MeanX1, MeanY1 = np.average(PointsArray1,0)
                 MeanX2, MeanY2 = np.average(PointsArray2,0)
                 
                 ## 2nd feature, horizontal distance ratio
                 HorDist = abs(MeanX1 - MeanX2)
                 if ((self.width + other.width)/2):
                         second_feature = HorDist/((self.width + other.width)/2)
                 else:
                         second_feature = 0
                 all_features.append(second_feature)
                 
                 ## 3rd feature, vertical distance ratio
                 VerDist = abs(MeanY1 - MeanY2)
                 if ((self.height + other.height)/2):
                         third_feature = VerDist/((self.height + other.height)/2)
                 else:
                         third_feature = 0
                 all_features.append(third_feature)

                 ## 4th feature, size difference ratio
                 Size1 = max(self.width,self.height)
                 Size2 = max(other.width,other.height)
                 SizeDif = abs(Size1 - Size2)
                 if ((Size1 + Size2)/2):
                         forth_feature = SizeDif/((Size1 + Size2)/2)
                 else:
                         forth_feature = 0
                 all_features.append(forth_feature)


                 ## the 5-10 features are from or similar to Winkler's papers, but details are missing how to calculate these features

                 ## 5th feature horizontal overlap ratio
                 mins, maxs = self.extents
                 o_mins, o_maxs = other.extents
                 if((mins[0] < o_maxs[0]) and (maxs[0] > o_mins[0])):
                         HorOverlap = min(maxs[0],o_maxs[0]) - max(mins[0],o_mins[0])
                 else:
                         HorOverlap = 0
                 if ((self.width + other.width)/2):
                         fifth_feature = HorOverlap/((self.width + other.width)/2)
                 else:
                         fifth_feature = 0
                 all_features.append(fifth_feature)

                 ## 6th feature vertical overlap ratio
                 if((mins[1] < o_maxs[1]) and (maxs[1] > o_mins[1])):
                         VerOverlap = min(maxs[1],o_maxs[1]) - max(mins[1],o_mins[1])
                 else:
                         VerOverlap = 0
                 if ((self.height + other.height)/2):
                         sixth_feature = VerOverlap/((self.height + other.height)/2)
                 else:
                         sixth_feature = 0
                 all_features.append(sixth_feature)

                 ## 7th feature distance between beginning points
                 BegDist = distance(self.points[0], other.points[0])
                 if (self.average_diag(other) * 2):
                         seventh_feature = BegDist/(self.average_diag(other) * 2)
                 else:
                         seventh_feature = 0
                 all_features.append(seventh_feature)

                 ## 8th feature distance between ending points
                 EndDist = distance(self.points[len(self.points)-1], other.points[len(other.points)-1])
                 if (self.average_diag(other) * 2):
                         eighth_feature = EndDist/(self.average_diag(other) * 2)
                 else:
                         eighth_feature = 0
                 all_features.append(eighth_feature)
                 
                 ## 9th feature distance between ending point of the first stroke and beginning point of the second stroke
                 EndBegDist = distance(self.points[len(self.points)-1], other.points[0])
                 if (self.average_diag(other) * 2):
                         ninth_feature = EndBegDist/(self.average_diag(other) * 2)
                 else:
                         ninth_feature = 0
                 all_features.append(ninth_feature)

                 ## 10th feature, parallelity of the two strokes (the angle of the two diagonals)
                 Vector1 = (maxs[0] - mins[0], maxs[1] - mins[1])
                 Vector2 = (o_maxs[0] - o_mins[0], o_maxs[1] - o_mins[1])
                 tenth_feature = angle(Vector1,Vector2)
                 all_features.append(tenth_feature)

                 ## 11-13 features are my thoughts
                 
                 ## 11th feature, distance between centers of the bounding box
                 if (self.average_diag(other) * 2):
                         eleventh_feature = self.center_distance(other) / (self.average_diag(other) * 2)
                 else:
                         eleventh_feature = 0
                 all_features.append(eleventh_feature)

                 ## 12th feature, distance between the averaged centers
                 AveCenDist = distance([MeanX1, MeanY1], [MeanX2, MeanY2])
                 if (self.average_diag(other) * 2):
                         twelveth_feature = AveCenDist / (self.average_diag(other) * 2)
                 else:
                         twelveth_feature = 0
                 all_features.append(twelveth_feature)

                 ## 13th feature, maximal distance between the two strokes/average of diagonal
                 if (self.average_diag(other) * 2):
                         thirteenth_feature = self.farest_distance(other) / (self.average_diag(other) * 2)
                 else:
                         thirteenth_feature = 0
                 all_features.append(thirteenth_feature)

                 ## 14th feature, overlapped area ratio, from MacLean's paper and the Chinese segmentation paper
                 AreaOverlap = HorOverlap * VerOverlap
                 if ((self.area + other.area)/2):
                         fourteenth_feature = AreaOverlap/((self.area + other.area)/2)
                 else:
                         fourteenth_feature = 0
                 all_features.append(fourteenth_feature)


                 ## 15, 16 feature are from Winkler's paper
                 ## 15th feature, horizontal offset between beginning points
                 BegHorDist = (-self.points[0][0] + other.points[0][0])
                 if ((self.width + other.width)/2):
                         fifteenth_feature = BegHorDist/((self.width + other.width)/2)
                 else:
                         fifteenth_feature = 0
                 all_features.append(fifteenth_feature)


                 ## 16th feature, horizontal offset between ending points
                 EndHorDist = (-self.points[len(self.points)-1][0] + other.points[len(other.points)-1][0])
                 if ((self.width + other.width)/2):
                         sixteenth_feature = EndHorDist/((self.width + other.width)/2)
                 else:
                         sixteenth_feature = 0
                 all_features.append(sixteenth_feature)

                 ## 17-20 features is my own thought
                 ## 17th feature, horizontal offset between ending point of the first stroke and beginning point of the second stroke
                 EndBegHorDist = (-self.points[len(self.points)-1][0] + other.points[0][0])
                 if ((self.width + other.width)/2):
                         seventeenth_feature = EndBegHorDist/((self.width + other.width)/2)
                 else:
                         seventeenth_feature = 0
                 all_features.append(seventeenth_feature)


                 ## 18th feature, the angle between the horizontal line and the line connecting the last point of the current stroke
                 ## and the first point of the next stroke
                 Vector3 = (-self.points[len(self.points)-1][0] + other.points[0][0], -self.points[len(self.points)-1][1] + other.points[0][1])
                 Vector4 = (1, 0)
                 eighteenth_feature = angle(Vector3,Vector4)
                 all_features.append(eighteenth_feature)


                 ## 19th feature, the angle between the line connecting the first point and last point of the current stroke
                 ## and the line connecting the first point of the next stroke
                 ## and the last point of the next stroke
                 Vector5 = (self.points[len(self.points)-1][0] - self.points[0][0], self.points[len(self.points)-1][1] - self.points[0][1])
                 Vector6 = (other.points[len(other.points)-1][0] - other.points[0][0], other.points[len(other.points)-1][1] - other.points[0][1])
                 nineteenth_feature = angle(Vector5,Vector6)
                 all_features.append(nineteenth_feature)


                 ## 20th feature, horizontal distance (between centers) ratio
                 HorCenDist = abs(other.center[0] - self.center[0])
                 if ((self.width + other.width)/2):
                         twentyth_feature = HorCenDist/((self.width + other.width)/2)
                 else:
                         twentyth_feature = 0
                 all_features.append(twentyth_feature)
                 
                 ## 21th feature, vertical distance (between centers) ratio
                 VerCenDist = abs(other.center[1] - self.center[1])
                 if ((self.height + other.height)/2):
                         twentyoneth_feature = VerCenDist/((self.height + other.height)/2)
                 else:
                         twentyoneth_feature = 0
                 all_features.append(twentyoneth_feature)

                 
                 

                 ## delete the NaN element in the feature vector
                 for i in range(len(all_features)):
                         if math.isnan(all_features[i]):
                                 all_features[i] = 0
                 return all_features


        ## context shape features from Ling's thesis
        def context_shape_features(self, other):
                context_shape = [[0 for x in xrange(5)] for x in xrange(12)]
                self_center = self.center
                self_diag = 2.0*self.half_diag
                point_number = 0 ## point number is the number of points in the context of the reference point
                total_number = len(self.points) + len(other.points)

                ##for the points in the current stroke
                for x in self.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1


                ##for the points in the next stroke
                for x in other.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1

                final_context_shape = []
                for i in range(12):
                        for j in range(5):
                                final_context_shape.append(context_shape[i][j]/float(point_number))

                ## delete the NaN element in the feature vector
                for i in range(len(final_context_shape)):
                         if math.isnan(final_context_shape[i]):
                                 final_context_shape[i] = 0
                return final_context_shape



        ## context shape features from Ling's thesis, but the length of radius is flexible to make the circle can
        ## cover but only can cover the two strokes in the stroke pair 
        def context_shape_features_1NN(self, other):
                context_shape = [[0 for x in xrange(5)] for x in xrange(12)]
                self_center = self.center
                self_diag = 0.0
                point_number = 0 ## point number is the number of points in the context of the reference point
                total_number = len(self.points) + len(other.points)


                for x in self.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance > self_diag:
                                self_diag = temp_distance


                for x in other.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance > self_diag:
                                self_diag = temp_distance

                ##for the points in the current stroke
                for x in self.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1


                ##for the points in the next stroke
                for x in other.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1

                final_context_shape = []
                for i in range(12):
                        for j in range(5):
                                final_context_shape.append(context_shape[i][j]/float(point_number))

                ## delete the NaN element in the feature vector

                for i in range(len(final_context_shape)):
                         if math.isnan(final_context_shape[i]):
                                 final_context_shape[i] = 0
                return final_context_shape                
                

        def bb_intersects(self, other):
                mins, maxs = self.extents
                o_mins, o_maxs = other.extents
                return (mins[0] <= o_maxs[0]) and (maxs[0] >= o_mins[0]) and (mins[1] <= o_maxs[1]) and (maxs[1] >= o_mins[1])
                
        def intersects(self, other):
                for s1, s2 in zip(self.points, self.points[1:]):
                        for o1, o2 in zip(other.points, other.points[1:]):
                                if s1 == o1 or s1 == o2 or s2 == o1 or s2 == o2:
                                        return True
                                v1 = np.cross(vect(s1, o1), vect(s1, s2))
                                v2 = np.cross(vect(s1, o2), vect(s1, s2))
                                if v1[2] * v2[2] < 0:
                                        w1 = np.cross(vect(o1, s1), vect(o1, o2))
                                        w2 = np.cross(vect(o1, s2), vect(o1, o2))
                                        if w1[2] * w2[2] < 0:
                                                return True
                return False

## dot product
def dotproduct(v1,v2):
        return sum((a*b) for a,b in zip(v1,v2))
            
## module of vector
def length(v):
        return math.sqrt(dotproduct(v,v))

## angle between two vectors
def angle(v1,v2):
        if (length(v1) * length(v2)):

                if dotproduct(v1,v2)/(length(v1) * length(v2)) >= 1.0:
                        return math.acos(1.0)
                elif dotproduct(v1,v2)/(length(v1) * length(v2)) <= -1.0:
                        return math.acos(-1.0)
                else:
                        return math.acos(dotproduct(v1,v2)/(length(v1) * length(v2)))
        else:
                return 0
     
      

def vect(a, b):
        return np.array([b[0] - a[0], b[1] - a[1], 0])
        
        
def count_nonadjacent_strokes(path):
        count = 0
        for filename in os.listdir(path):
                if os.path.splitext(filename)[1] == '.inkml':
                        print(filename)
                        eq = Equation.from_inkml(os.path.join(path, filename))
                        for s in eq.segments_truth:
                                m = min(s.strokes)
                                l = len(s.strokes)
                                for i in range(m + 1, m + l):
                                        if i not in s:
                                                count += 1
                                                print(s)
        print('%d segments contain non-adjacent strokes' % count)
        
def split_stats(stats, mapping, partition, filtering=None, filter_outliers=True):
    if filter_outliers:
        stats = filter(lambda x: not x.is_wide and not x.is_dot, stats)
    if filtering:
        stats = filter(filtering, stats)
    fst = map(mapping, filter(partition, stats))
    snd = map(mapping, filter(lambda x: not partition(x), stats))
    return fst, snd
    
def show_hist(data):
    if not isinstance(data[0], collections.Iterable):
        data = [data]
    for d in data:
        plt.hist(d, bins=500 / 3, range=(0,1), alpha=0.7)
    plt.show()

def test_dots(path):
        dot_symbols = ['i', '\\sin', '!', '\\lim', '\\div', 'j', '\\ldots']
        files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.inkml']
        for t in [0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]:
                num_correct, num_total, num_truth = 0, 0, 0
                for i, filename in enumerate(files):
                        eq = Equation.from_inkml(os.path.join(path, filename))
                        dots = eq.get_dots(t)
                        symbs = map(lambda x: eq.segment_truth_for_stroke(x.id).symbol, dots)
                        num_correct += len(filter(lambda x: x in dot_symbols, symbs))
                        num_total += len(symbs)
                        num_truth += len(filter(lambda x: x.symbol in dot_symbols, eq.segments_truth))
                print('\nthreshold: %.2f' % (t))
                print('precision:\t%f\t(%d/%d)' % (float(num_correct) / num_total, num_correct, num_total))
                print('recall:\t\t%f\t(%d/%d)' % (float(num_correct) / num_truth, num_correct, num_truth))
        
        
def get_distance_stats(path):
        Stat = collections.namedtuple('Stat', ['together', 'closest_distance', 'center_distance', 'is_wide', 'is_dot', 'symbols'])
        stats = []
        for filename in os.listdir(path):
                if os.path.splitext(filename)[1] == '.inkml':
                        print(filename)
                        eq = Equation.from_inkml(os.path.join(path, filename))
                        
                        wides = eq.get_wide_strokes()
                        dots = eq.get_dots()
                        
                        pairs = zip(eq.strokes.values(), eq.strokes.values()[1:])
                        for s1, s2 in pairs:
                                try:
                                        seg1 = eq.segment_truth_for_stroke(s1.id)
                                        seg2 = eq.segment_truth_for_stroke(s2.id)
                                        av_diag = s1.average_diag(s2)
                                        closest_dist = s1.closest_distance(s2) / av_diag
                                        center_dist = s1.center_distance(s2) / av_diag
                                        is_wide = s1 in wides or s2 in wides
                                        is_dot = s1 in dots or s2 in dots
                                        symbols = (seg1.symbol, seg2.symbol)
                                        stats.append(Stat(seg1 == seg2, closest_dist, center_dist, is_wide, is_dot, symbols))
                                        
                                except KeyError:
                                        pass
        if __name__ == '__main__':
                plt.hist([x.closest_distance for x in stats if x.together and x.closest_distance < 5], alpha=0.7, bins=500)
                plt.hist([x.closest_distance for x in stats if not x.together and x.closest_distance < 5], alpha=0.7, bins=500)
                plt.show()
                
        return stats
        

class symbol_candidate(object):
        def __init__(self, symbol_label, stroke_list, points, top_three):
                self.symbol_label = symbol_label
                self.stroke_list = stroke_list
                self.points = points
                self.top_three = top_three

        @property
        def extents(self):
                if not (hasattr(self, '_mins') and hasattr(self, '_maxs')):
                        mins = list(self.points[0])
                        maxs = list(self.points[0])
                
                        for point in self.points:
                                mins = zipwith(min, mins, point)
                                maxs = zipwith(max, maxs, point)
                        self._mins, self._maxs = tuple(mins), tuple(maxs)
                return self._mins, self._maxs
                
        @property
        def center(self):
                mins, maxs = self.extents
                box = zipwith(lambda x,y: abs(x - y), mins, maxs)
                return zipwith(lambda m,b: m + (float(b) / 2), mins, box)

        def closest_distance(self, other):
                closest_dis = 1000000
                for i in range(len(self.stroke_list)):
                        for j in range(len(other.stroke_list)):
                                temp_closest_dis = self.stroke_list[i].closest_distance(other.stroke_list[j])
                                if temp_closest_dis < closest_dis:
                                        closest_dis = temp_closest_dis
                return closest_dis


## build a directory 'f' if it doesn't exists
def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


## the entrance program for CROHME 2014
def DPRL_CROHME2014(path, output_path):

        ensure_dir(output_path)
        cur_dir = os.getcwd()
        CROHMELib_dir = os.path.join(cur_dir, 'crohmelib/bin/')
        os.chdir(CROHMELib_dir)
        ##convert the format of inkml files
        subprocess.call(['./normalizeSymbols', path])
        os.chdir(cur_dir)

        files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.inkml']
        for i, filename in enumerate(files):
                print('%s (%d/%d)' % (filename, i + 1, len(files)))
                ## read the inkml file
                eq = Equation.from_inkml(os.path.join(path, filename))
                O_eq = copy.deepcopy(eq)
                
                ## get segmentation results
                eq.lei_CROHME2013_segment()

                ## get classification results
                symbol_candidate_list = []       
                for seg in eq.segments:
                        stroke_list = []
                        points = []
                        for stro in seg.strokes:
                                ## the original stroke data
                                stroke_list.append(O_eq.strokes[stro])
                                points = points + O_eq.strokes[stro].points

                        trace_list = []
                        for j in range(len(stroke_list)):
                                one_trace = (stroke_list[j].id, stroke_list[j].points)
                                trace_list.append(one_trace)
                                
                        ## do the classification for a given symbol candidate
                        symbol_label = KENNY_CLASSIFIER.mostProbableLabel(KENNY_CLASSIFIER.classify(copy.deepcopy(trace_list)))[0]
                        ## get the top N classification result based on the classification confidence
                        top_N_num = 3
                        top_three = KENNY_CLASSIFIER.topNLabels(KENNY_CLASSIFIER.classify(copy.deepcopy(trace_list)), top_N_num)

                        ## deal with several special symbols to make the lable for them to be consistent
                        if symbol_label == '\\cdot':
                                symbol_label = '.'
                        if symbol_label == '\\tg':
                                symbol_label = '\\tan'

                        for j in range(top_N_num):
                                if top_three[j][0] == '\\cdot':
                                        top_three[j] = tuple(['.', top_three[j][1]])
                                if top_three[j][0] == '\\tg':
                                        top_three[j] = tuple(['\\tan', top_three[j][1]])
                                        
                        ## get one symbol candidate with its top 3 classification results and confidence
                        one_symbol_candidate = symbol_candidate(symbol_label, stroke_list, points, top_three)
                        symbol_candidate_list.append(one_symbol_candidate)

                ##parsing
                relation_tree = CROHME2013_parsing_MST(symbol_candidate_list)

                ## get the LG file
                ## for the expression data, which has the stroke index not beginning from 0
                ## debug for CROHME 2014
                write_LG_with_id_map(filename, symbol_candidate_list, relation_tree, eq.id_map)
                LG_name = os.path.join(cur_dir, filename[:len(filename)-6] + '.lg')
                if os.path.exists(LG_name):
                        shutil.copyfile(LG_name, os.path.join(output_path, filename[:len(filename)-6] + '.lg'))
                        os.remove(LG_name)
                        

##write the recognition result as a LG
def write_LG(file_name, symbol_candidate_list, relation_tree):
        dirname = os.getcwd()
        new_file_name = file_name[:len(file_name)-6]
        with open(os.path.join(dirname, new_file_name + '.lg'), 'w') as sfile:
                sfile.write('# IUD, ' + new_file_name + '\n')
                sfile.write('# Nodes:' + '\n')
                ## write for the Ns
                symbol_num = len(symbol_candidate_list)
                for i in range(symbol_num):
                        for j in range(len(symbol_candidate_list[i].stroke_list)):
                                sfile.write('N, ')
                                sfile.write('%d' % symbol_candidate_list[i].stroke_list[j].id)
                                sfile.write(', ' + symbol_candidate_list[i].symbol_label + ', ')
                                sfile.write('%f' % 1.0)
                                sfile.write('\n')

                ## write for the Es
                sfile.write('\n')
                sfile.write('# Edges:' + '\n')

                ## write for the Es with relationship with '*'
                for i in range(symbol_num):
                        stroke_num = len(symbol_candidate_list[i].stroke_list)
                        if stroke_num > 1:
                                stroke_id_list = []
                                for j in range(stroke_num):
                                        stroke_id_list.append(symbol_candidate_list[i].stroke_list[j].id)

                                for j in range(stroke_num):
                                        for k in range(stroke_num):
                                                if j!=k:
                                                        stroke_id_1 = stroke_id_list[j]
                                                        stroke_id_2 = stroke_id_list[k]
                                                        sfile.write('E, ')
                                                        sfile.write('%d' % stroke_id_1)
                                                        sfile.write(', ')
                                                        sfile.write('%d' % stroke_id_2)
                                                        sfile.write(', ')
                                                        sfile.write('*' + ', ')
                                                        sfile.write('%f' % 1.0)
                                                        sfile.write('\n')
                                        
                        else:
                                continue
                        

                ## write for the Es with relationship but '*'
                edge_num = len(relation_tree)
                for i in range(edge_num):
                        one_edge = relation_tree[i]
                        if one_edge[2] == 'Same':
                                continue
                        else:
                                PC_R = one_edge[2] ## parent child relation
                                symbol_list_1 = one_edge[0]
                                symbol_list_2 = one_edge[1]
                                for j in range(len(symbol_list_1)):
                                        parent_symbol = symbol_list_1[j]
                                        for k in range(len(symbol_list_2)):
                                                child_symbol = symbol_list_2[k]
                                                for l in range(len(parent_symbol.stroke_list)):
                                                        S1 = parent_symbol.stroke_list[l]
                                                        for m in range(len(child_symbol.stroke_list)):
                                                                S2 = child_symbol.stroke_list[m]
                                                                sfile.write('E, ')
                                                                sfile.write('%d' % S1.id)
                                                                sfile.write(', ')
                                                                sfile.write('%d' % S2.id)
                                                                sfile.write(', ')
##                                                                print "PC_R", PC_R
                                                                sfile.write(PC_R + ', ')
                                                                sfile.write('%f' % 1.0)
                                                                sfile.write('\n')
                                                

                               

## for the expression data, which has the stroke index not beginning from 0
## debug for CROHME 2014
def write_LG_with_id_map(file_name, symbol_candidate_list, relation_tree, id_map):
        dirname = os.getcwd()
        new_file_name = file_name[:len(file_name)-6]
        with open(os.path.join(dirname, new_file_name + '.lg'), 'w') as sfile:
                sfile.write('# IUD, ' + new_file_name + '\n')
                sfile.write('# Nodes:' + '\n')
                ## write for the Ns
                symbol_num = len(symbol_candidate_list)
                for i in range(symbol_num):
                        for j in range(len(symbol_candidate_list[i].stroke_list)):
                                sfile.write('N, ')
                                ## for the expression data, which has the stroke index not beginning from 0
                                ## debug for CROHME 2014
                                sfile.write('%d' % id_map[symbol_candidate_list[i].stroke_list[j].id])
                                sfile.write(', ' + symbol_candidate_list[i].symbol_label + ', ')
                                sfile.write('%f' % 1.0)
                                sfile.write('\n')

                ## write for the Es
                sfile.write('\n')
                sfile.write('# Edges:' + '\n')

                ## write for the Es with relationship with '*'
                for i in range(symbol_num):
                        stroke_num = len(symbol_candidate_list[i].stroke_list)
                        if stroke_num > 1:
                                stroke_id_list = []
                                for j in range(stroke_num):
                                        stroke_id_list.append(symbol_candidate_list[i].stroke_list[j].id)

                                for j in range(stroke_num):
                                        for k in range(stroke_num):
                                                if j!=k:
                                                        stroke_id_1 = stroke_id_list[j]
                                                        stroke_id_2 = stroke_id_list[k]
                                                        sfile.write('E, ')
                                                        sfile.write('%d' % id_map[stroke_id_1])
                                                        sfile.write(', ')
                                                        sfile.write('%d' % id_map[stroke_id_2])
                                                        sfile.write(', ')
                                                        sfile.write('*' + ', ')
                                                        sfile.write('%f' % 1.0)
                                                        sfile.write('\n')
                                        
                        else:
                                continue
                        

                ## write for the Es with relationship but '*'
                edge_num = len(relation_tree)
                for i in range(edge_num):
                        one_edge = relation_tree[i]
                        if one_edge[2] == 'Same':
                                continue
                        else:
                                PC_R = one_edge[2] ## parent child relation
                                symbol_list_1 = one_edge[0]
                                symbol_list_2 = one_edge[1]
                                for j in range(len(symbol_list_1)):
                                        parent_symbol = symbol_list_1[j]
                                        for k in range(len(symbol_list_2)):
                                                child_symbol = symbol_list_2[k]
                                                for l in range(len(parent_symbol.stroke_list)):
                                                        S1 = parent_symbol.stroke_list[l]
                                                        for m in range(len(child_symbol.stroke_list)):
                                                                S2 = child_symbol.stroke_list[m]
                                                                sfile.write('E, ')
                                                                sfile.write('%d' % id_map[S1.id])
                                                                sfile.write(', ')
                                                                sfile.write('%d' % id_map[S2.id])
                                                                sfile.write(', ')
                                                                sfile.write(PC_R + ', ')
                                                                sfile.write('%f' % 1.0)
                                                                sfile.write('\n')                        


## write symbols for paco for spatial relationship classification
def write_Paco_symbol(one_symbol, n):
        dirname = os.getcwd()
        with open(os.path.join(dirname, 'sym' + str(n)), 'w') as sfile:
                sfile.write(one_symbol.symbol_label + '\n')
                sfile.write('%d \n' % len(one_symbol.stroke_list))
                for i in range(len(one_symbol.stroke_list)):
                        sfile.write('%d \n' % len(one_symbol.stroke_list[i].points))
                        for p in one_symbol.stroke_list[i].points:
                                sfile.write('%f %f\n' % p)



## get Paco relationships for two symbols
def get_Paco_R(sym_1,sym_2):
        write_Paco_symbol(sym_1, 1)
        write_Paco_symbol(sym_2, 2)

        Paco_string = subprocess.check_output(['./layout', 'sym1', 'sym2',  'MODhbp.svm',  'MAT.pca'])
        space_location = find(Paco_string, ' ')
        newline_location = find(Paco_string, '\n')
        R_score = []

        for i in range(3):
                R_score.append(float(Paco_string[space_location[i]:newline_location[i]]))

        R_list = ['Sub', 'R', 'Sup']
        return R_list[R_score.index(max(R_score))]


## get Paco relationships for two symbols by using MST
def get_Paco_R_MST(sym_1,sym_2):
        if sym_2.symbol_label == '-' or sym_2.symbol_label == '\\frac' or sym_2.symbol_label == '\\sum' or sym_2.symbol_label == '\\lim' or sym_2.symbol_label == '\\sqrt':
                S_R = get_Paco_R(sym_1, sym_2)
        else:
                classification_score = []
                relation_score = []
                s_relation = []
                for i in range(len(sym_2.top_three)):
                        classification_score.append(sym_2.top_three[i][1])
                        sym_2.symbol_label = sym_2.top_three[i][0]
                        one_relation = get_Paco_R_Score(sym_1, sym_2)
                        relation_score.append(one_relation[0])
                        s_relation.append(one_relation[1])

                total_score = []
                for i in range(len(sym_2.top_three)):
                        total_score.append(classification_score[i]*relation_score[i])

                max_index = total_score.index(max(total_score))
                sym_2.symbol_label = sym_2.top_three[max_index][0]
                S_R = s_relation[max_index]

        return S_R
                
                



## get Paco relationships and scores for two symbols
def get_Paco_R_Score(sym_1,sym_2):
##def get_Paco_R():
        write_Paco_symbol(sym_1, 1)
        write_Paco_symbol(sym_2, 2)

        Paco_string = subprocess.check_output(['./layout', 'sym1', 'sym2',  'MODhbp.svm',  'MAT.pca'])
        space_location = find(Paco_string, ' ')
        newline_location = find(Paco_string, '\n')
        R_score = []

        for i in range(3):
                R_score.append(float(Paco_string[space_location[i]:newline_location[i]]))

        R_list = ['Sub', 'R', 'Sup']
        return [max(R_score), R_list[R_score.index(max(R_score))]]
        



## get vertical overlap ratio between two symbols
def get_VOR(sym_1, sym_2):
        height_1 = sym_1.extents[1][1] - sym_1.extents[0][1]
        height_2 = sym_2.extents[1][1] - sym_2.extents[0][1]
        min_height = min(height_1, height_2)
        
        ## get the vertical overlap
        mins, maxs = sym_1.extents
        o_mins, o_maxs = sym_2.extents

        VerOverlap = 0.0
        if((mins[1] < o_maxs[1]) and (maxs[1] > o_mins[1])):
                VerOverlap = min(maxs[1],o_maxs[1]) - max(mins[1],o_mins[1])
        else:
                VerOverlap = 0.0

        VOR = 0.0
        if min_height:
                VOR = VerOverlap/min_height
        else:
                VOR = 1.0

        return VOR
        
def CROHME2013_parsing_MST(symbol_candidate_list):
        relation_tree = []
        symbol_num = len(symbol_candidate_list)
        left_side = []
        for i in range(symbol_num):
                left_side.append(symbol_candidate_list[i].extents[0][0])

        sorted_index = sorted(range(symbol_num), key=lambda k: left_side[k])
        symbol_use =[0 for x in xrange(int(symbol_num))]## to indidate the symbol is used or not
        baseline_symbol = []
        sub_expression_list = []


        ## find the dominant symbol and its symbol group
        for i in range(symbol_num):
                the_label = symbol_candidate_list[sorted_index[i]].symbol_label
                dominant_symbol = symbol_candidate_list[sorted_index[i]]
                
                if (the_label == '-' or the_label == '\\frac' or the_label == '\\sum' or the_label == '\\lim') and (symbol_use[sorted_index[i]] == 0):
                        baseline_symbol.append(symbol_candidate_list[sorted_index[i]])
                        symbol_use[sorted_index[i]] = 1
                        below_symbol_list = []
                        above_symbol_list = []
                        for j in range(symbol_num):
                                if symbol_use[sorted_index[j]] == 0:
                                        
                                        child_symbol = symbol_candidate_list[sorted_index[j]]
                                        if is_above(dominant_symbol, child_symbol): ## child_symbol is above dominant symbol
                                                above_symbol_list.append(child_symbol)
                                                symbol_use[sorted_index[j]] = 1
                                        if is_below(dominant_symbol, child_symbol):
                                                below_symbol_list.append(child_symbol)
                                                symbol_use[sorted_index[j]] = 1

                        if len(above_symbol_list) > 0:
                                one_sub_expression = sub_expression(above_symbol_list, dominant_symbol, 'A')
                                sub_expression_list.append(one_sub_expression)

                        if len(below_symbol_list) > 0:
                                one_sub_expression = sub_expression(below_symbol_list, dominant_symbol, 'B')
                                sub_expression_list.append(one_sub_expression)

                if (the_label == '\\sqrt') and (symbol_use[sorted_index[i]] == 0):
                        baseline_symbol.append(symbol_candidate_list[sorted_index[i]])
                        symbol_use[sorted_index[i]] = 1
                        inside_symbol_list = []
                        for j in range(symbol_num):
                                if symbol_use[sorted_index[j]] == 0:               
                                        child_symbol = symbol_candidate_list[sorted_index[j]]
                                        if is_inside(dominant_symbol, child_symbol): ## child_symbol is inside dominant symbol
                                                inside_symbol_list.append(child_symbol)
                                                symbol_use[sorted_index[j]] = 1

                        if len(inside_symbol_list) > 0:
                                one_sub_expression = sub_expression(inside_symbol_list, dominant_symbol, 'I')
                                sub_expression_list.append(one_sub_expression)
        
        ## find the symbols which are not dominant symbols and not within the dominant symbol groups
        for i in range(symbol_num):
                if symbol_use[sorted_index[i]] == 0:
                        baseline_symbol.append(symbol_candidate_list[sorted_index[i]])
                        symbol_use[sorted_index[i]] = 1


        ## for the symbols in the current baseline, get the relationships between each successive symbols from right to left
        baseline_symbol_num = len(baseline_symbol)
        baseline_left_side = []
        for i in range(baseline_symbol_num):
                baseline_left_side.append(baseline_symbol[i].extents[0][0])

        baseline_sorted_index = sorted(range(baseline_symbol_num), key=lambda k: baseline_left_side[k])

##        baseline_R = []
        if baseline_symbol_num > 1:
                reference_sym = baseline_symbol[baseline_sorted_index[0]]
                reference_index = 0
                to_find_right_sym = 1
                while to_find_right_sym == 1:
                        if reference_index+1< baseline_symbol_num:## the reference symbol is not the rightmost symbol
                                        right_sym_found = 0
                                        for i in range(reference_index+1 ,baseline_symbol_num):
                                                temp_R = get_Paco_R_MST(reference_sym, baseline_symbol[baseline_sorted_index[i]])
                                                if temp_R == 'R':## the right symbol is found
                                                        right_sym_found = 1
                                                        one_edge = [[reference_sym], [baseline_symbol[baseline_sorted_index[i]]], temp_R]
                                                        relation_tree.append(one_edge)
                                                        sym_gap = i - reference_index
                                                        if sym_gap > 1: ## means there are symbols between in the two adjacent symbols
                                                                for j in range(reference_index, i-1):
                                                                        ## add the edge based on the score
                                                                        relation_1 = get_Paco_R_Score(reference_sym, baseline_symbol[baseline_sorted_index[j+1]])# relation_1 is [score, R]
                                                                        relation_2 = get_Paco_R_Score(baseline_symbol[baseline_sorted_index[j]], baseline_symbol[baseline_sorted_index[j+1]])
                                                                        if relation_1[0] >= relation_2[0]:
                                                                                temp_R = get_Paco_R_MST(reference_sym, baseline_symbol[baseline_sorted_index[j+1]])
                                                                                one_edge = [[reference_sym], [baseline_symbol[baseline_sorted_index[j+1]]], temp_R]
                                                                                relation_tree.append(one_edge)
                                                                        else:
                                                                                temp_R = get_Paco_R_MST(baseline_symbol[baseline_sorted_index[j]], baseline_symbol[baseline_sorted_index[j+1]])
                                                                                one_edge = [[baseline_symbol[baseline_sorted_index[j]]], [baseline_symbol[baseline_sorted_index[j+1]]], temp_R]
                                                                                relation_tree.append(one_edge)
                    
                                                        reference_index = i
                                                        reference_sym = baseline_symbol[baseline_sorted_index[i]]
                                                        break
                                                
                                        if right_sym_found == 0: ##the right symbol is not found
                                                to_find_right_sym =0
                                                for j in range(reference_index, baseline_symbol_num-1):
                                                        ## add the edge based on the score
                                                        relation_1 = get_Paco_R_Score(reference_sym, baseline_symbol[baseline_sorted_index[j+1]])# relation_1 is [score, R]
                                                        relation_2 = get_Paco_R_Score(baseline_symbol[baseline_sorted_index[j]], baseline_symbol[baseline_sorted_index[j+1]])
                                                        if relation_1[0] >= relation_2[0]:
                                                                temp_R = get_Paco_R_MST(reference_sym, baseline_symbol[baseline_sorted_index[j+1]])
                                                                one_edge = [[reference_sym], [baseline_symbol[baseline_sorted_index[j+1]]], temp_R]
                                                                relation_tree.append(one_edge)
  
                                                        else:
                                                                temp_R = get_Paco_R_MST(baseline_symbol[baseline_sorted_index[j]], baseline_symbol[baseline_sorted_index[j+1]])
                                                                one_edge = [[baseline_symbol[baseline_sorted_index[j]]], [baseline_symbol[baseline_sorted_index[j+1]]], temp_R]
                                                                relation_tree.append(one_edge)
                                    
                        else:
                                to_find_right_sym = 0
                                
                
                
        else:
                one_edge = [[baseline_symbol[0]], [baseline_symbol[0]], 'Same']
                relation_tree.append(one_edge)
                        
        if len(sub_expression_list)>0:
                for i in range(len(sub_expression_list)):
                        one_edge = [[sub_expression_list[i].dominant_symbol], sub_expression_list[i].symbol_list, sub_expression_list[i].spatial_r]
                        relation_tree.append(one_edge)
                        ## parsing recursively
                        sub_relation_tree = CROHME2013_parsing_MST(sub_expression_list[i].symbol_list)
                        relation_tree = relation_tree + sub_relation_tree

        return relation_tree


class sub_expression(object):
        def __init__(self, symbol_list, dominant_symbol, spatial_r):
                self.symbol_list = symbol_list
                self.dominant_symbol = dominant_symbol
                self.spatial_r = spatial_r


def is_above(sym_1, sym_2): ## sym_2 is above sym_1, the center of sym_2 is above sym_1, they have horizontal overlap, the lowest point of sym_2 is higer than the lowest point of sym_1
        mins, maxs = sym_1.extents
        o_mins, o_maxs = sym_2.extents
        if sym_2.center[1] < sym_1.center[1] and (mins[0] <= o_maxs[0]) and (maxs[0] >= o_mins[0]) and maxs[1] > o_maxs[1] :
                return 1
        else:
                return 0


def is_below(sym_1, sym_2): 
        mins, maxs = sym_1.extents
        o_mins, o_maxs = sym_2.extents
        if sym_2.center[1] > sym_1.center[1] and (mins[0] <= o_maxs[0]) and (maxs[0] >= o_mins[0]) and mins[1] < o_mins[1] :
                return 1
        else:
                return 0
        
def is_inside(dominant_symbol, child_symbol):
        if child_symbol.center[0] >= dominant_symbol.extents[0][0] and child_symbol.center[0] <= dominant_symbol.extents[1][0] and child_symbol.center[1] >= dominant_symbol.extents[0][1] and child_symbol.center[1] <= dominant_symbol.extents[1][1]:
                return 1
        else:
                return 0

def get_MST(symbol_candidate_list):
        symbol_num = len(symbol_candidate_list)
        symbol_dis = [[0.0 for x in xrange(int(symbol_num))] for x in xrange(int(symbol_num))]
        for i in range(symbol_num):
                for j in range(symbol_num):
                        if j > i:
                                symbol_dis[i][j] =  symbol_candidate_list[i].closest_distance(symbol_candidate_list[j])

        symbol_dis_matrix = csr_matrix(symbol_dis)
        Tcsr = minimum_spanning_tree(symbol_dis_matrix)
        MST = Tcsr.toarray()
        return MST

def get_two_CC(O_current_stroke, O_next_stroke):

        id_1 = O_current_stroke.id
        id_2 = O_next_stroke.id
        points_1 = O_current_stroke.points
        points_2 = O_next_stroke.points
        trace_1 = (id_1, points_1)
        trace_2 = (id_2, points_2)
        symbol_1 = [trace_1]
        symbol_2 = [trace_1, trace_2]
        CC_1 = KENNY_CLASSIFIER.classify(copy.deepcopy(symbol_1))
        CC_2 = KENNY_CLASSIFIER.classify(copy.deepcopy(symbol_2))
        two_CC = CC_1 + CC_2
        return two_CC


## preprocess equation, delete the repeated points, normalizing, smoothing and resampling
def equation_preprocessing(raw_equation):
        ## normalizing the equation
        normalize_eq = equation_normalizing(raw_equation)
        
        temp_eq = normalize_eq
        for i in range(len(temp_eq.strokes)):
                current_stroke = temp_eq.strokes[i]
                clean_stroke = delete_duplicate_point(current_stroke)
                smooth_stroke = smoothing(clean_stroke)
                resample_stroke = resampling(smooth_stroke)
                second_clean_stroke = delete_duplicate_point(resample_stroke)
                temp_eq.strokes[i] = second_clean_stroke

        return temp_eq




## normalizing the equation, the y coordinate range is [0,200] while preserving the width height aspect ratil
def equation_normalizing(raw_equation):
        normalize_eq = raw_equation
        
        ## get the min_x, min_y, max_x, max_y
        all_mins = []
        all_maxs = []
        for i in range(len(raw_equation.strokes)):
                current_stroke = raw_equation.strokes[i]
                mins, maxs = current_stroke.extents
                all_mins.append(mins)
                all_maxs.append(maxs)
                
        all_mins = np.array(all_mins)
        all_maxs = np.array(all_maxs)
        min_x = min(all_mins[:,0])
        min_y = min(all_mins[:,1])
        max_x = max(all_maxs[:,0])
        max_y = max(all_maxs[:,1])
        eq_width = max_x - min_x
        eq_height = max_y - min_y
        normalize_ratio = 200.0/eq_height

        ## get the new points after normaliztion
        for i in range(len(normalize_eq.strokes)):
                current_stroke = normalize_eq.strokes[i]
                points = current_stroke.points
                points = np.array(points)
                points[:,0] = points[:,0] - min_x
                points[:,1] = points[:,1] - min_y
                normalize_points = points*normalize_ratio
                final_points = []
                for j in range(len(normalize_points)):
                        final_points.append(tuple(normalize_points[j,:]))
##                normalize_eq.strokes[i].points = final_points
                normalize_eq.strokes[i] = Stroke(normalize_eq.strokes[i].id,final_points)

        return normalize_eq
                
                
         
                
## delete the duplicate point
def delete_duplicate_point(current_stroke):
        points = current_stroke.points
        if len(points)>1:
                new_points = []
                new_points.append(points[0])
                for i in range(len(points)-1):
                        current_point = points[i]
                        next_point = points[i+1]
                        if current_point != next_point:
                                new_points.append(next_point)
        else:
                new_points = points

        current_stroke = Stroke(current_stroke.id, new_points)
        return current_stroke

## smoothing, replace the point's coordinate by the average of the previous point, the current point and the following point
def smoothing(clean_stroke):
        points = clean_stroke.points
        if len(points)>2:
                new_points = []
                new_points.append(points[0])
                for i in range(len(points)-2):
                        pre_point = points[i]
                        current_point = points[i+1]
                        next_point = points[i+2]
                        smooth_point = (np.array(pre_point) + np.array(current_point) + np.array(next_point))/3.0
                        new_points.append(tuple(smooth_point))
                new_points.append(points[len(points)-1])
        else:
                new_points = points

        clean_stroke = Stroke(clean_stroke.id, new_points)
        return clean_stroke


## resampling
def resampling(smooth_stroke):
        points = smooth_stroke.points
        if len(points)>1:
                new_points = []
                dl = 3.125e-1
                ## line rendering
                for i in range(len(points)-1):
                        current_point = points[i]
                        next_point = points[i+1]
                        dx = next_point[0] - current_point[0]
                        dy = next_point[1] - current_point[1]
                        l = 0.0
                        while(l<1.0):
                                x = current_point[0] + dx*l
                                y = current_point[1] + dy*l
                                l+=dl
                                new_points.append((x, y))
                new_points.append(points[len(points)-1])
        else:
                new_points = points

        final_new_points = []

        for i in range(len(new_points)):
                final_new_points.append((round(new_points[i][0]), round(new_points[i][1])))

        smooth_stroke = Stroke(smooth_stroke.id, final_new_points)
        return smooth_stroke


## get the 3NN background shape context feature
## current_stroke is the reference stroke
## the radius will be flexible to can but only can can its 3 nearest neighbor
def get_3NN_background_scf(eq, current_stroke):
        context_shape = [[0 for x in xrange(5)] for x in xrange(12)]
        self_center = current_stroke.center ## reference_center
        self_diag = 0 ## reference_diag
        point_number = 0
        total_number = 0
        neighbor_num = 4# include the current stroke itself

        ## find the 3NN, plus the current stroke, it will be 4 strokes
        NN_id = []
        if len(eq.strokes)>neighbor_num:
                Nearest_Dis = []
                for i in range(len(eq.strokes)):
                        one_stroke = eq.strokes[i]
                        one_Nearest_Dis = current_stroke.closest_distance(one_stroke)
                        Nearest_Dis.append(one_Nearest_Dis)
                s = Nearest_Dis
                sorted_index = sorted(range(len(s)), key = lambda k:s[k])
                NN_id = sorted_index[:neighbor_num]
        else:
                ## if the stroke number is less than 5, then it will include all the strokes
                NN_id = range(len(eq.strokes))

        ## find the self_diag
        temp_self_diag = 0.0
        for i in range(len(NN_id)):
                one_stroke = eq.strokes[NN_id[i]]
                for x in one_stroke.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance > temp_self_diag:
                                temp_self_diag = temp_distance

        self_diag = temp_self_diag
                
        ## calculate the feature 
        for j in range(len(NN_id)):
                now_stroke = eq.strokes[NN_id[j]]
                total_number += len(now_stroke.points)
                for x in now_stroke.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1

        final_context_shape = []
        for i in range(12):
                for j in range(5):
                        if point_number > 0:
                                final_context_shape.append(context_shape[i][j]/float(point_number))
                        else:
                                final_context_shape.append(0)

        ## delete the NaN element in the feature vector
        for i in range(len(final_context_shape)):
                 if math.isnan(final_context_shape[i]):
                         final_context_shape[i] = 0
        return final_context_shape     

                


## get the global shape context feature
## current_stroke is the reference stroke
def get_global_scf(eq, current_stroke):
        context_shape = [[0 for x in xrange(5)] for x in xrange(12)]
        self_center = current_stroke.center ## reference_center
        self_diag = 0.0 ## reference_diag
        point_number = 0
        total_number = 0

        ## get the radius of the circle which can cover the whole expression
        for j in range(len(eq.strokes)):
                now_stroke = eq.strokes[j]
                for x in now_stroke.points:
                        temp_distance = distance(x,self_center)
                        if temp_distance > self_diag:
                                self_diag = temp_distance
                   
        for j in range(len(eq.strokes)):
                now_stroke = eq.strokes[j]
                total_number += len(now_stroke.points)
                for x in now_stroke.points:
                        
                        temp_distance = distance(x,self_center)
                        if temp_distance <= self_diag:
                                point_number += 1
                                v1 = (x[0] - self_center[0], x[1] - self_center[1])
                                v2 = (1,0)## horizontal line
                                temp_angle = angle(v1,v2) ## the angle is in radian
                                if self_diag == 0:
                                        distance_ratio = 0
                                else:
                                        distance_ratio = temp_distance/self_diag
                                if distance_ratio<=1.0/16:
                                        col_index = 0
                                elif 1.0/16<distance_ratio<=1.0/8:
                                        col_index = 1
                                elif 1.0/8<distance_ratio<=1.0/4:
                                        col_index = 2
                                elif 1.0/4<distance_ratio<=1.0/2:
                                        col_index = 3
                                else:
                                        col_index = 4
                                        
                                ## the polor coordinated is divided into 12 parts based on the angle, each part contain pi/6, and math.atan(1) is pi/4
                                angle_ratio = 1.5*(temp_angle/math.atan(1))

                                if v1[1]>=0:
                                        row_index = math.floor(angle_ratio)
                                else:
                                        row_index = 11 - math.floor(angle_ratio)

                                context_shape[int(row_index)][int(col_index)]+=1

        final_context_shape = []
        for i in range(12):
                for j in range(5):
                        final_context_shape.append(context_shape[i][j]/float(point_number))

        ## delete the NaN element in the feature vector

        for i in range(len(final_context_shape)):
                 if math.isnan(final_context_shape[i]):
                         final_context_shape[i] = 0
        return final_context_shape    



## find all indexes of a char in a string
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

                        
if __name__ == '__main__':
       
        if len(sys.argv) < 3 or sys.argv[1] not in globals():
                usage_statement = [
                        'Usage: python segmentation.py <command>',
                        'where command is:',
                        'DPRL_CROHME2014 <input_path> <output_path>'
                        ]
                sys.exit('\n\t'.join(usage_statement))

        # first argument is the function name - call it, passing in the rest of the arguments
        globals()[sys.argv[1]](*sys.argv[2:])
