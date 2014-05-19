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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include "symbol.h"

#define PI 3.14159265

using namespace std;

float **loadPCA( char *path );
void psm(symbol *s1, symbol *s2, svm_node *sample, float **pca);

int main(int argc, char* argv[]) {

  if( argc != 5 ) {
    fprintf(stderr, "Usage: %s sym1 sym2 model.svm pca.mat\n", argv[0]);
    return -1;
  }

  //Load symbols
  symbol s1(argv[1]);
  symbol s2(argv[2]);

  //Load model
  SVMclass model(argv[3]);
  svm_node sample[41];
  double probs[3];

  //Load PCA matrix
  float **pca = loadPCA( argv[4] );

  //Compute Geometric Features
  s1.BBfeatures( &s2, sample );

  //Compute Shape-based features
  psm(&s1, &s2, sample, pca);

  model.classify(sample, probs);

  for(int i=0; i<3; i++) {
    //WARNING!! Index-class depends on the model file, but not readed, using the values
    //for the RIT crohme system with Horizonta, Subscript and Superscript such that in
    //the training set the subscript appears in first place.
    if( i==0 ) printf("Sub %.4f\n", probs[i]);
    else if( i==1 ) printf("Hor %.4f\n", probs[i]);
    else if( i==2 ) printf("Sup %.4f\n", probs[i]);
  }

  return 0;
}


float **loadPCA( char *path ) {

  FILE *fd = fopen(path, "r");
  if( !fd ) {
    fprintf(stderr, "Error loading file '%s'\n", path);
    exit(-1);
  }
  
  int R,C;
  fscanf(fd, "%d %d", &R, &C);
  
  if( R<=0 || C<=0 ) {
    fprintf(stderr, "Error: Wrong dimensions of PCA matrix\n");
    exit(-1);
  }
  
  //Read PCA matrix values
  float **pca = new float*[R];
  for(int i=0; i<R; i++) {
    pca[i] = new float[C];
    for(int j=0; j<C; j++)
      fscanf(fd, "%f", &pca[i][j]);
  }
  
  fclose(fd);

  return pca;
}


void psm(symbol *s1, symbol *s2, svm_node *sample, float **pca) {
  int NP1, NP2, NPother;
  float cx1=0,cy1=0;
  float cx2=0,cy2=0;

  //Compute centroid of symbol i
  NP1=0;
  for(int s=0; s<s1->NS; s++) {
    NP1 += s1->strks[s][0].x;
    for(int i=1; i<=s1->strks[s][0].x; i++) {
      cx1 += s1->strks[s][i].x;
      cy1 += s1->strks[s][i].y;
    }
  }
  cx1 /= NP1;
  cy1 /= NP1;

  //Compute centroid of symbol j
  //Compute centroid of symbol i
  NP2=0;
  for(int s=0; s<s2->NS; s++) {
    NP2 += s2->strks[s][0].x;
    for(int i=1; i<=s2->strks[s][0].x; i++) {
      cx2 += s2->strks[s][i].x;
      cy2 += s2->strks[s][i].y;
    }
  }
  cx2 /= NP2;
  cy2 /= NP2;

#ifdef VERBOSE
  fprintf(stderr, "cen-s1 (%.2f,%.2f)\n", cx1, cy1);
  fprintf(stderr, "cen-s2 (%.2f,%.2f)\n", cx2, cy2);
#endif 

  //Set the origin in middle point between c1 and c2
  cx1 = (cx1+cx2)/2.0;
  cy1 = (cy1+cy2)/2.0;

  //Compute maximum radio
  float mrx, mry; //Maximum radius of the shape
  float d2 = -1;

  for(int s=0; s<s1->NS; s++) {
    for(int i=1; i<=s1->strks[s][0].x; i++) {
      float dist = (s1->strks[s][i].x - cx1)*(s1->strks[s][i].x - cx1)
	+ (s1->strks[s][i].y - cy1)*(s1->strks[s][i].y - cy1);
      
      if( dist > d2 ) {
	mrx = s1->strks[s][i].x;
	mry = s1->strks[s][i].y;
	d2 = dist;
      }
    }
  }

  for(int s=0; s<s2->NS; s++) {
    for(int i=1; i<=s2->strks[s][0].x; i++) {
      float dist = (s2->strks[s][i].x - cx1)*(s2->strks[s][i].x - cx1)
	+ (s2->strks[s][i].y - cy1)*(s2->strks[s][i].y - cy1);
    
      if( dist > d2 ) {
	mrx = s2->strks[s][i].x;
	mry = s2->strks[s][i].y;
	d2 = dist;
      }
    }
  }

#ifdef VERBOSE
  fprintf(stderr, "cen-psm(%.2f,%.2f)\n", cx1, cy1);
  fprintf(stderr, "mr     (%.2f,%.2f)\n", mrx, mry);
#endif

  //15 distances and 20 angles for polar shape matrix representation
  int N=15;
  int M=20;

  //Compute Polar Shape Matrix (PSM)
  d2 = sqrt(d2)/N;
  float arc = 360.0/M;
  
  int *PSM = new int[N*M];
  for(int i=0; i<N*M; i++)
    PSM[i] = 0;
  
  //Symbol-i
  for(int s=0; s<s1->NS; s++) {
    for(int i=1; i<=s1->strks[s][0].x; i++) {
      float dis = sqrt((s1->strks[s][i].x - cx1)*(s1->strks[s][i].x - cx1) + 
		       (s1->strks[s][i].y - cy1)*(s1->strks[s][i].y - cy1));
      
      float ang = atan2( s1->strks[s][i].y - cy1, s1->strks[s][i].x - cx1 )*180.0/PI;
      if( ang < 0.0 ) ang += 360;
      
      int pn = (int)(dis/d2);
      int pm = (int)(ang/arc);
      
      if( pn == N ) pn--;
      if( pm == M ) pm--;
      
      PSM[ pn*M + pm ]--; 
      
#ifdef VERBOSE
      fprintf(stderr, "S1: (%f,%f) %f %f => %d %d\n", 
              s1->strks[s][i].x, s1->strks[s][i].y, sqrt(dis), ang, pn, pm);
#endif

    }
  }

  //Symbol-j
  for(int s=0; s<s2->NS; s++) {
    for(int i=1; i<=s2->strks[s][0].x; i++) {
      float dis = sqrt((s2->strks[s][i].x - cx1)*(s2->strks[s][i].x - cx1) + 
		       (s2->strks[s][i].y - cy1)*(s2->strks[s][i].y - cy1));
      
      float ang = atan2( s2->strks[s][i].y - cy1, s2->strks[s][i].x - cx1 )*180.0/PI;
      if( ang < 0.0 ) ang += 360;
      
      int pn = (int)(dis/d2);
      int pm = (int)(ang/arc);
      
      if( pn == N ) pn--;
      if( pm == M ) pm--;
      
      PSM[ pn*M + pm ]++;

      //If there is the same name of points of sets in this bin
      if( PSM[ pn*M + pm ] == 0 )
        PSM[ pn*M + pm ] = 1; //Set the second one

#ifdef VERBOSE
      fprintf(stderr, "S2: (%f,%f) %f %f => %d %d\n", 
              s2->strks[s][i].x, s2->strks[s][i].y, sqrt(dis), ang, pn, pm);
#endif
    }
  }

  //Save features using only values -1/0/+1
  /*for(int i=0; i<N; i++) {
    for(int j=0; j<M; j++) {
      if(      PSM[i*M+j] < 0 ) PSM[i*M+j] = -1;
      else if( PSM[i*M+j] > 0 ) PSM[i*M+j] =  1;
      
      printf(" %d", PSM[i*M+j]);
    }
  }
  printf("\n");*/

  for(int i=0; i<N*M; i++) {
    if(      PSM[i] < 0 ) PSM[i] = -1;
    else if( PSM[i] > 0 ) PSM[i] =  1;
  }
  
  //Project N*M features into 31 using PCA
  for(int k=0; k<31; k++) {
    float aux = 0.0;
    for(int j=0; j<N*M; j++)
      aux += PSM[j] * pca[k][j];
    sample[9+k].index = 10+k;
    sample[9+k].value = aux;
  }
  sample[40].index = -1; //end mark
}
