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

import sys
import xml.etree.ElementTree as ET
from symbol_classifier import *

#=====================================================================
#  Submission for task1 of CROHME 2014
#
#  Created by:
#      - Kenny Davila (Apr 13, 2014)
#
#======================================================================

#the current XML namespace prefix...
INKML_NAMESPACE = '{http://www.w3.org/2003/InkML}'
#the number of results to show
TOP_N = 10

def load_inkml_traces(file_name):
    #first load the tree...
    tree = ET.parse(file_name)
    root = tree.getroot()

    #get all annotations...
    annotations = root.findall(INKML_NAMESPACE + 'annotation')
    #find UI annotation...
    UI = None
    for annotation in annotations:
        if "type" in annotation.attrib:
            if annotation.attrib["type"] == "UI":
                UI = annotation.text
                break

    #extract all the traces first...
    traces_list = []
    for trace in root.findall(INKML_NAMESPACE + 'trace'):
        #text contains all points as string, parse them and put them
        #into a list of tuples...
        points_s = trace.text.split(",");
        points_f = []
        for p_s in points_s:
            #split again...
            coords_s = p_s.split()
            #add...
            points_f.append( (float(coords_s[0]), float(coords_s[1])) )

        trace_id = int(trace.attrib['id'])

        trace_info = (trace_id, points_f)
        traces_list.append(trace_info)

    return UI, traces_list

def main():
    #usage check
    if len(sys.argv) != 2:
        print("Usage: python task1_submission.py list_filename")
        print("Where")
        print("\tlist_filename\t= Path to list of .inkml files to process")
        return


    classifier = SymbolClassifier("./task1_classifier.rsvm", "./task1_parameters.dat")

    #now, load the file with the parameters
    list_filename = sys.argv[1]
    list_file = open(list_filename, 'r')
    all_filenames = list_file.readlines()
    list_file.close()

    for filename in all_filenames:
        #process inkml file....
        UI, traces = load_inkml_traces(filename.strip())

        #now, classify...
        confidences = classifier.classify(traces)

        top_classes = classifier.topNLabels(confidences, TOP_N)

        class_str = ""
        scores_str = ""
        for label, score in top_classes:
            class_str += "," + label
            scores_str += "," + str(round(score, 4))

        print(UI + class_str)
        print("scores" + scores_str)
        #print(top_10)

main()
