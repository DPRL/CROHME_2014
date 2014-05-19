/*
    DPRL CROHME 2014
    Copyright (c) 2013-2014 Lei Hu, Kenny Davila, Francisco Alvaro, Richard Zanibbi

    This file is part of DPRL CROHME 2014.

    DPRL CROHME 2014 is free software: 
    you can redistribute it and/or modify it under the terms of the GNU 
    General Public License as published by the Free Software Foundation, 
    either version 3 of the License, or (at your option) any later version.

    DPRL CROHME 2014 is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DPRL CROHME 2014.  
    If not, see <http://www.gnu.org/licenses/>.

    Contact:
        - Lei Hu: lei.hu@rit.edu
        - Kenny Davila: kxd7282@rit.edu
        - Francisco Alvaro: falvaro@dsic.upv.es
        - Richard Zanibbi: rlaz@cs.rit.edu 
*/

#ifndef _SYMBOL_
#define _SYMBOL_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include "svm-classifier.h"

using namespace std;

struct point{
  float x;
  float y;
};

struct symbol{
  point **strks;
  int NS;
  string label;

  //Bounding Box (bx,by)-(bs,bt)
  float bx, by, bs, bt;
  float cen; //Vertical center
  char type;

  symbol(char *path);
  ~symbol();

  void computeBB();
  void BBfeatures( symbol *sym, svm_node *sample );
  void setType();
};

#endif
