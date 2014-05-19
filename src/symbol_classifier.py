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

import cPickle
import numpy as np
from mathSymbol import *

class SymbolClassifier:
    def __init__(self, training_filename, mapping_filename):
        #load classifier parameters...
        training_file = open(training_filename, 'rb')
        self.classifier = cPickle.load(training_file)
        training_file.close()

        #load classifier mapping and scaler
        mapping_file = open(mapping_filename, 'rb')
        self.classes_dict = cPickle.load(mapping_file)
        self.classes_list = cPickle.load(mapping_file)
        self.scaler = cPickle.load(mapping_file)
        mapping_file.close()

        print(len(self.classes_list))

        #by default...
        self.apply_scaler = True

    def createSymbol(self, trace_group):
        #first, create the traceInfo...
        traces = []
        for trace_id, points_f in trace_group:
            #create object...
            object_trace = TraceInfo(trace_id, points_f)

            #apply general trace pre processing...
            #1) first step of pre processing: Remove duplicated points
            object_trace.removeDuplicatedPoints()

            #Add points to the trace...
            object_trace.addMissingPoints()

            #Apply smoothing to the trace...
            object_trace.applySmoothing()

            #it should not ... but .....
            if object_trace.hasDuplicatedPoints():
                #...remove them! ....
                object_trace.removeDuplicatedPoints()

            traces.append(object_trace)

        #now create the symbol
        new_symbol = MathSymbol(0, traces, '{Unknown}')

        #normalization ...
        new_symbol.normalize()

        return new_symbol

    def classify(self, trace_group):
        #create the symbol object...
        symbol = self.createSymbol(trace_group)

        #get raw features
        features = symbol.getFeatures()

        #convert them to numpy matrix...
        matrix_features = np.matrix([features])
        if self.apply_scaler:
            matrix_features = self.scaler.transform(matrix_features)

        predicted = self.classifier.predict_proba(matrix_features)

        confidences = np.array(predicted).reshape(-1).tolist()

        return confidences

    def mostProbableLabel(self, confidences):
        most_probable = 0
        for i in range(1, len(confidences)):
            if confidences[i] > confidences[most_probable]:
                most_probable = i

        return self.classes_list[most_probable], confidences[most_probable]

    def topNLabels(self, confidences, n_top):

        all_scores = [(confidences[i], self.classes_list[i]) for i in xrange(len(self.classes_list))]
        sorted_scores = sorted(all_scores, reverse=True)

        return [(class_label, class_confidence) for class_confidence, class_label in sorted_scores[0:n_top]]



