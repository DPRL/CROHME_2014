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

#include "symbol.h"

using namespace std;

symbol::symbol(char *path) {
  FILE *fd = fopen(path, "r");
  if( !fd ) {
    fprintf(stderr, "Error loading file '%s'\n", path);
    exit(-1);
  }

  //Symbol class
  char line[2048];
  fscanf(fd, "%s", line);
  label = line;

  setType();
  
  //Number of strokes
  fscanf(fd, "%d", &NS);
  strks = new point*[NS];

  for(int i=0; i<NS; i++) {
    //Number of points
    int NP;
    fscanf(fd, "%d", &NP);
    strks[i] = new point[NP+1];
    strks[i][0].x = NP;

    for(int j=1; j<=NP; j++)
      fscanf(fd, "%f %f", &strks[i][j].x, &strks[i][j].y);
  }

  computeBB();
}

symbol::~symbol() {
  for(int i=0; i<NS; i++)
    delete[] strks[i];
  delete[] strks;
}

void symbol::setType() {
  type = 'n';
  if( !label.compare("b") )      type = 'a';
  else if( !label.compare("0") ) type = 'm';
  else if( !label.compare("1") ) type = 'm';
  else if( !label.compare("2") ) type = 'm';
  else if( !label.compare("3") ) type = 'm';
  else if( !label.compare("4") ) type = 'm';
  else if( !label.compare("5") ) type = 'm';
  else if( !label.compare("6") ) type = 'm';
  else if( !label.compare("7") ) type = 'm';
  else if( !label.compare("8") ) type = 'm';
  else if( !label.compare("9") ) type = 'm';
  else if( !label.compare("A") ) type = 'm';
  else if( !label.compare("B") ) type = 'm';
  else if( !label.compare("\\beta") ) type = 'd';
  else if( !label.compare("C") ) type = 'm';
  else if( !label.compare("d") ) type = 'a';
  else if( !label.compare("E") ) type = 'm';
  else if( !label.compare("f") ) type = 'a';
  else if( !label.compare("F") ) type = 'm';
  else if( !label.compare("g") ) type = 'd';
  else if( !label.compare("G") ) type = 'm';
  else if( !label.compare("h") ) type = 'a';
  else if( !label.compare("H") ) type = 'm';
  else if( !label.compare("I") ) type = 'm';
  else if( !label.compare("j") ) type = 'd';
  else if( !label.compare("l") ) type = 'a';
  else if( !label.compare("L") ) type = 'm';
  else if( !label.compare("\\lambda") ) type = 'a';
  else if( !label.compare("\\lim") ) type = 'a';
  else if( !label.compare("lpar") ) type = 'm';
  else if( !label.compare("M") ) type = 'm';
  else if( !label.compare("\\mu") ) type = 'd';
  else if( !label.compare("N") ) type = 'm';
  else if( !label.compare("p") ) type = 'd';
  else if( !label.compare("P") ) type = 'm';
  else if( !label.compare("q") ) type = 'd';
  else if( !label.compare("R") ) type = 'm';
  else if( !label.compare("rpar") ) type = 'm';
  else if( !label.compare("S") ) type = 'm';
  else if( !label.compare("\\sqrt") ) type = 'm';
  else if( !label.compare("t") ) type = 'a';
  else if( !label.compare("T") ) type = 'm';
  else if( !label.compare("\\tan") ) type = 'a';
  else if( !label.compare("\\tg") ) type = 'a';
  else if( !label.compare("V") ) type = 'm';
  else if( !label.compare("X") ) type = 'm';
  else if( !label.compare("y") ) type = 'd';
  else if( !label.compare("Y") ) type = 'm';
}

void symbol::computeBB() {
  //Compute Bounding box and centroid
  bx = by =  FLT_MAX;
  bs = bt = -FLT_MAX;
  cen = 0;
  
  int n=0;
  for(int i=0; i<NS; i++)
    for(int j=1; j<=strks[i][0].x; j++) {

      if( strks[i][j].x < bx ) bx = strks[i][j].x;
      if( strks[i][j].x > bs ) bs = strks[i][j].x;
      if( strks[i][j].y < by ) by = strks[i][j].y;
      if( strks[i][j].y > bt ) bt = strks[i][j].y;
      
      cen += strks[i][j].y;

      n++;
    }
  
  cen /= n;
  
  if( type == 'a' )
    cen = (cen + bt)/2.0;
  else if( type == 'd' )
    cen = (cen + by)/2.0;
  else if( type == 'm' )
    cen = (by + bt)/2.0;
}

void symbol::BBfeatures( symbol *sym, svm_node *sample ) {
  float h1 = bt - by + 1;
  float h2 = sym->bt - sym->by + 1;
  
  h1 = max(bt,sym->bt) - min(by,sym->by);

  //Compute 9 spatial classification features
  sample[0].index = 1;
  sample[0].value = h2/h1;
  sample[1].index = 2;
  sample[1].value = (cen - sym->cen)/h1;
  sample[2].index = 3;
  sample[2].value = ((bs - bx)/2.0 - (sym->bs - sym->bx)/2.0)/h1;
  sample[3].index = 4;
  sample[3].value = (sym->bx - bs)/h1;
  sample[4].index = 5;
  sample[4].value = (sym->bx - bx)/h1;
  sample[5].index = 6;
  sample[5].value = (sym->bs - bs)/h1;
  sample[6].index = 7;
  sample[6].value = (sym->by - bt)/h1;
  sample[7].index = 8;
  sample[7].value = (sym->by - by)/h1;
  sample[8].index = 9;
  sample[8].value = (sym->bt - bt)/h1;
  //sample[9].index = -1; //End mark
}
