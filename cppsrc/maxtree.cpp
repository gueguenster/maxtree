/*
cCOPYRIGHT

All contributions by L. Gueguen:
Copyright (c) 2016
All rights reserved.


LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "maxtree.h"
#include <math.h>
#include <map>
#include <algorithm>
#include <set>
#include <iostream>
#include <iterator>

using namespace std;

template < typename PixelType >
MaxTree< PixelType >::MaxTree()
{ connectivity=4;
}

template < typename PixelType >
MaxTree< PixelType >::MaxTree(const vector<PixelType> & imarray, unsigned int w, unsigned int h ){
	readim(imarray,w,h);
	connectivity=4;
}

template < typename PixelType >
MaxTree< PixelType >::MaxTree(const vector<PixelType> & imarray, unsigned int w, unsigned int h , int conn){
	readim(imarray,w,h);
	setConnectivity(conn);
}

template < typename PixelType >
MaxTree< PixelType >::MaxTree(PixelType* imarray, unsigned int w, unsigned int h){
	unsigned int tot = w*h;
	vector< PixelType > image(tot);
	for(unsigned int i=0;i<tot;++i)
		image[i] = imarray[i];
	readim(image,h,w);
	connectivity = 4;
}

template < typename PixelType >
void MaxTree< PixelType >::_resetMaps2(ui curHeader){

	auto s = pixelheader2cc.find(curHeader);

	if(s==end(pixelheader2cc)){ // the header is not yet here put it on
		if(parent[curHeader]!=curHeader){ // recurse only if not root
			_resetMaps2(parent[curHeader]);
		}
		cc2pixelheader.push_back(curHeader);
		pixelheader2cc[curHeader] = cc2pixelheader.size()-1;

	}

	return ;
}


template < typename PixelType >
MaxTree< PixelType >::MaxTree(const vector<ui> & par, const vector<PixelType> & di, unsigned int w, unsigned int h){
	width = w;
	height = h;
	nbpixels = w*h;

	parent = par;
	diff = di;

	//recompute the header2
	for(ui p=0;p<nbpixels;p++){
		if(diff[p]){ // we are dealing with header
			_resetMaps2(p);
		}
	}

	//recompute the image
	computeImage();
}

template < typename PixelType >
MaxTree< PixelType >::MaxTree(const vector<ui> & par, const vector<PixelType> & di, const vector<ui> & cc2pixelhdr,
	    unsigned int w, unsigned int h){
	width = w;
	height = h;
	nbpixels = w*h;

	parent = par;
	diff = di;

    cc2pixelheader = cc2pixelhdr;
    for(ui cc_idx=0; cc_idx<cc2pixelheader.size(); ++cc_idx){
        pixelheader2cc[cc2pixelheader[cc_idx]] = cc_idx;
    }

	//recompute the image
	computeImage();
}

template < typename PixelType >
MaxTree< PixelType >::MaxTree(unsigned int dummy, unsigned int * retained, unsigned int lr){
	width = retained[0];
	height = retained[1];
	nbpixels = width*height;
	parent.resize(nbpixels);
	diff.resize(nbpixels);
	for(ui i=0; i< nbpixels; ++i){
		parent[i] = retained[2*i+2];
		diff[i] = retained[2*i+1+2];
	}

	//recompute the header2
	for(ui p=0;p<nbpixels;p++){
		if(diff[p]){ // we are dealing with header
			_resetMaps2(p);
		}
	}

	//recompute the image
	computeImage();
}

template < typename PixelType >
void MaxTree< PixelType >::readim(const vector<PixelType> & imarray, unsigned int w, unsigned int h ){
	width = w;
	height = h;
	nbpixels = w*h;

	im = imarray;

	parent.resize(nbpixels);
	diff.resize(nbpixels);

}

template <typename PixelType >
ui MaxTree< PixelType >::HQ_first(PixelType h){
	//return the first element of the queue at level h, and remove it
	//typedef typename map< PixelType, queue< ui > >::iterator thequeueath = hierarchicalqueues.find(h);;
	auto thequeueath = hierarchicalqueues.find(h);
	ui p = thequeueath->second.front();
	thequeueath->second.pop();
	return p;
}

template <typename PixelType >
void MaxTree< PixelType >::HQ_add(PixelType h, ui p){
	// add the position p at the end of the queue at level h
	hierarchicalqueues[h].push(p); // hierarchicalqueues[h] gets created if does not exists
}

template <typename PixelType >
bool MaxTree< PixelType >::HQ_noempty(PixelType h){
	// determine if the queue is not empty
	auto thequeueath = hierarchicalqueues.find(h);
	bool isempty =  (thequeueath == end(hierarchicalqueues)) || ( thequeueath->second.empty() );
	return !isempty;
}

template <typename PixelType >
vector < ui > MaxTree< PixelType >::GetNeighbors(ui p){
   // compute 4-neighbors
   vector < ui > neighbors;
   ui x = p%width;
   if (x<(width-1)) 								neighbors.push_back(p+1);		// (x+1,y)
   if (p>=width) 									neighbors.push_back(p-width);	// (x,y-1)
   if (x>0)											neighbors.push_back(p-1);		// (x-1,y)
   if ((p + width) <nbpixels) 						neighbors.push_back(p+width);	// (x,y+1)
   if (connectivity>4){
	   if ((x<(width-1)) && (p>=width) ) 			neighbors.push_back(p-width+1);	// (x+1,y-1)
	   if ((x>0) && (p>=width) ) 					neighbors.push_back(p-width-1);	// (x-1,y-1)
	   if ((x<(width-1)) && (p+width < nbpixels)) 	neighbors.push_back(p+width+1);	// (x+1,y+1)
	   if ((x>0) && (p+width < nbpixels)) 			neighbors.push_back(p+width-1);	// (x-1,y+1)
   }
   return neighbors;
}

template < typename PixelType >
void MaxTree< PixelType >::compute(){
   // Find pixel m which has the lowest intensity l in the image
   auto argmin = min_element(begin(im), end(im));
   ui argmin_p = distance(begin(im), argmin);
   PixelType themin = im[argmin_p];

   //visited array
   vector < bool > hasBeenVisisted(nbpixels);
   fill(begin(hasBeenVisisted),end(hasBeenVisisted),false);

   // mapping maintaining the leaders
   map < PixelType, ui > leaders;
   leaders[themin]=argmin_p;

   // Add pixel argmin_p to the queue
   HQ_add(themin,argmin_p);
   hasBeenVisisted[argmin_p]=true;

   _compute(hasBeenVisisted, leaders, themin);
   parent[argmin_p] = argmin_p;
   diff[argmin_p] = themin;
}

template < typename PixelType >
PixelType MaxTree< PixelType >::_compute(vector< bool > & hasBeenVisited,
		map < PixelType, ui > & leaders,
		PixelType h){

	// main big while
	while( HQ_noempty(h) ){

		ui p = HQ_first(h);						//pull the pixel at level h
		vector < ui > neighs = GetNeighbors(p); // get its neighbors

		for (auto qq=begin(neighs); qq!=end(neighs); ++qq){ //iterate on the neighbors
			ui q = *qq; // position of the neighbor

			if( !hasBeenVisited[q] ){
				hasBeenVisited[q] = true;
				PixelType fq = im[q];

				auto lerofq = leaders.find(fq); // try to find the leader at fq
				if (lerofq==end(leaders)){ // check is there is not any leader at level fq
					leaders[fq] = q; // q becomes the leader
				}else{
					parent[q] = leaders[fq]; // take the current leader
				}

				HQ_add(fq,q); //add the current pixel to the queue at its level

				while(fq>h){			 // explore the upper levels
					fq = _compute(hasBeenVisited,leaders, fq);
				}
			}
		}
	}

	// find the parent of the current leader
	auto curLead = leaders.find(h);
	if(curLead != end(leaders)){

		ui posOfLead = curLead->second;
		auto prevLead = leaders.find(h);
		prevLead --;
		PixelType m = prevLead->first;

		parent[posOfLead] = leaders[m];
		diff[posOfLead] = h-m;

		ui ccIndex = cc2pixelheader.size(); // compute the index of the CC, next in cc2pixelheader
		pixelheader2cc[posOfLead] = ccIndex ; // update the map to have the pixel leader to point to the CC position
		cc2pixelheader.push_back(posOfLead);  // put the pixel leader at the end of the current cc2pixelheader

		leaders.erase(curLead); // remove the leader at level h

		return m;
	}else{

		return 0;
	}

}


template<typename PixelType>
inline PixelType MaxTree<PixelType>::_computeImage(ui curHeader,
		map<ui, PixelType> & pixelheader2value) {

	ui locparent = parent[curHeader];

	auto f = pixelheader2value.find(curHeader);

	if(f==end(pixelheader2value)){ // if the header already is not in the map
		PixelType value;

		if(locparent==curHeader) //if the header is the root
			value = diff[curHeader];
		else
			value = diff[curHeader] + _computeImage(locparent,pixelheader2value);

		pixelheader2value[curHeader] = value;
		return value;

	}else{					// if the header is in the map
		return f->second;
	}
}

template<typename PixelType>
void MaxTree<PixelType>::computeImage() {
	if (im.size()==0){
		im.resize(nbpixels);
		map < ui, PixelType > pixelheader2value;
		for (auto pixelheader: pixelheader2cc){
			_computeImage(pixelheader.first, pixelheader2value);
		}

		for(ui i=0; i<nbpixels;++i){
			if(diff[i]){ // that a header pixel
				im[i]=pixelheader2value[i];
			}else{
				im[i]=pixelheader2value[parent[i]];
			}
		}

	}

}



template < typename PixelType >
void MaxTree< PixelType >::_print(){
	cout << "IMage: " << endl;
	for(ui p=0; p< nbpixels;++p){
		if(!(p%width))
			cout << endl;
		cout << (float)im[p] << "\t";
	}
	cout << endl << endl;

	cout << "Parent: " << endl;
	for(ui p=0; p< nbpixels;++p){
		if(!(p%width))
			cout << endl;
		if(parent[p]==p)
			cout<< "(*,*)" << "\t";
		else
			cout<< "(" << parent[p]%width << "," << parent[p]/width << ")" << "\t";
	}
	cout << endl << endl;

	cout << "diff: " << endl;
	for(ui p=0; p< nbpixels;++p){
		if(!(p%width))
			cout << endl;
		cout<< (float)diff[p] << "\t";
	}
	cout << endl << endl;

	cout << " cc2pixelheader : " << endl;
	int count=0;
	for(auto it : cc2pixelheader){
		cout << count << "->(" << it%width <<","<< it/width << ")" << " ";
		count++;
	}
	cout << endl << endl;

	cout << " pixelheader2cc : " << endl;
	for(auto it : pixelheader2cc)
		cout << "(" << it.first%width <<","<< it.first/width << ")->"<<it.second << "  ";
	cout << endl << endl;

}

/********************** Filtering functions **************************/
template<typename PixelType>
inline float MaxTree<PixelType>::_filter(
		ui curHeader,
		map< ui, float > & pixelheader2value,
		const map< ui, float> & isretained) {

	ui locparent = parent[curHeader];

	auto f = pixelheader2value.find(curHeader);

	if(f==end(pixelheader2value)){ // if the header is not in the map
		float value = isretained.find(curHeader)->second * (float) diff[curHeader];

		if(locparent!=curHeader) //if the header is not the  root
			value += _filter(locparent,pixelheader2value, isretained);

		pixelheader2value[curHeader] = value;
		return value;

	}else{					// if the header is in the map
		return f->second;
	}
}

template<typename PixelType>
inline map <ui, float > MaxTree<PixelType>::_filterall(
		const map< ui, float> & isretained){

	map < ui , float > pixelheader2value;
	for (auto pixelheader: pixelheader2cc){
		_filter(pixelheader.first, pixelheader2value, isretained);
	}
	return pixelheader2value;
}

template<typename PixelType>
inline void MaxTree<PixelType>::_filterallpixels(
		vector <float> & res,
		const map< ui, float> & isretained){

	map < ui , float > pixelheader2value = _filterall(isretained);
	for(ui i=0; i<nbpixels;++i){
		if(diff[i]){ // that a header pixel
					res[i]=pixelheader2value[i];
		}else{
					res[i]=pixelheader2value[parent[i]];
		}
	}
	return;
}

template < typename PixelType >
vector <PixelType> MaxTree<PixelType>::filter( const vector < ui > & retained ){
	// retained is a vector of the CC indices to remap
	vector < PixelType > res(nbpixels);

	// populate is retained vector
	map< ui, float> isretained;
	for(auto header : pixelheader2cc){
		isretained[header.first] = 0.0;
	}
	for(auto cc : retained){
		if (cc < cc2pixelheader.size())
				isretained[ cc2pixelheader[ cc ] ] = 1.0;
	}

	//launch the procedure
	vector < float > tmp(nbpixels);
	_filterallpixels(tmp, isretained);
	for(ui p=0;p<nbpixels;++p){
		res[p] = (PixelType) tmp[p];
	}

	//
	return res;
}

//vector <ui>		   filter( const vector <ui> & retained);       // map the largest labels

template < typename PixelType >
vector <PixelType> MaxTree<PixelType>::filter( const vector < bool > & retained ){
	// retained is a vector of length = nb of CC to remap
	vector < PixelType > res(nbpixels);

	// populate is retained vector
	map< ui, float> isretained;
	for(auto header : pixelheader2cc){
		isretained[header.first] = 0.0;
	}
	for(ui cc=0; cc < retained.size(); ++cc){
		if (( retained[cc] ) && ( cc < cc2pixelheader.size() ))
			isretained[ cc2pixelheader[ cc ] ] = 1.0;
	}

	//launch the procedure
	vector < float > tmp(nbpixels);
	_filterallpixels(tmp, isretained);
	for(ui p=0;p<nbpixels;++p){
		res[p] = tmp[p];
	}

	//
	return res;
}

template < typename PixelType >
vector <float>     MaxTree<PixelType>::filter( const vector < pair< ui, float> > & retained ){
	// retained is a vector of CC indices with score to remap
	vector < float > res(nbpixels);

	// populate is retained vector
	map< ui, float> isretained;
	for(auto header : pixelheader2cc){
		isretained[header.first] = 0.0;
	}
	for(auto cc_score : retained){
		if (cc_score.first < cc2pixelheader.size())
			isretained[ cc2pixelheader[ cc_score.first ] ] = cc_score.second;
	}

	//launch the procedure
	_filterallpixels(res, isretained);

	//
	return res;
}
template < typename PixelType >
void MaxTree<PixelType>::filter_swig( unsigned int * retained, unsigned int lr,
		float* score, unsigned int ls,
		float ** out, unsigned int* w, unsigned int *h){

	// place retained and score into a vector of pair
	vector< pair < ui, float > > retainedIndices(lr);
	for(ui i=0;i<lr;++i){
		retainedIndices[i].first = retained[i];
		retainedIndices[i].second = score[i];
	}

	// do the filtering
	vector <float> res = filter(retainedIndices);

	// copy the filtering into the output
	*w = height;
	*h = width;
	*out = new float[nbpixels];
	for(ui j=0;j<nbpixels;++j)
		out[0][j] = res[j];
}
template < typename PixelType >
void MaxTree<PixelType>::filter_swig( unsigned int * retained, unsigned int lr,
		PixelType ** out, unsigned int* w, unsigned int *h){

	// place retained and score into a vector of pair
	vector< ui > retainedIndices(lr);
	for(ui i=0;i<lr;++i)
		retainedIndices[i] = retained[i];

	// do the filtering
	vector < PixelType > res = filter(retainedIndices);

	// copy the filtering into the output
	*w = height;
	*h = width;
	*out = new PixelType[nbpixels];
	for(ui j=0;j<nbpixels;++j)
		out[0][j] = res[j];
}

/********************** Covering CC **********************************/
template < typename PixelType >
void MaxTree<PixelType>::_addcover(ui p, set < ui > & covering){
	if(diff[p]){ // header
		ui cc = pixelheader2cc[p];
		if(covering.find(cc) == end(covering)){
			if(p!=parent[p])
				_addcover(parent[p],covering);
			covering.insert(cc);
		}
	}else{
		_addcover( parent[p], covering);
	}
}

template < typename PixelType >
vector < ui > MaxTree<PixelType>::coveringCC( const vector < ui > & pixelToCover, const vector < ui >& pixelNotToCover){
	// returns the CC covering the given pixels, but not the others


	set < ui > covering;
	for(auto p : pixelToCover){
		_addcover(p,covering);
	}

	set < ui > nocovering;
	for(auto p : pixelNotToCover){
		_addcover(p,nocovering);
	}

	vector < ui > res(covering.size());
	auto it = set_difference (begin(covering), end(covering),
							 begin(nocovering), end(nocovering),
							 begin(res));

	res.resize(it-begin(res));

	return res;
}

template < typename PixelType >
vector < ui > MaxTree<PixelType>::coveringCC( const vector < ui > & pixelToCover){
	// returns the CC covering the given pixels
	vector < ui > NT(0);
	return coveringCC( pixelToCover, NT);
}

template < typename PixelType >
vector < ui > MaxTree<PixelType>::coveringCC_XY( const vector < ui > & pixelToCoverX, const vector < ui > & pixelToCoverY){
	// returns the CC covering the given pixels

	vector < ui > pNTCX(0);
	vector < ui > pNTCY(0);
	return coveringCC_XY(pixelToCoverX,pixelToCoverY,pNTCX,pNTCY);
}

template < typename PixelType >
vector < ui > MaxTree<PixelType>::coveringCC_XY( const vector < ui > & pixelToCoverX, const vector < ui > & pixelToCoverY,
						  const vector < ui > & pixelNotToCoverX, const vector < ui > & pixelNotToCoverY){

	vector < ui > pixelToCover(pixelToCoverX.size());
	vector < ui > pixelNotToCover(pixelNotToCoverX.size());
	for(unsigned int it=0; it< pixelToCoverX.size(); ++it ){
		pixelToCover[it] = pixelToCoverY[it]*width + pixelToCoverX[it];
	}

	for(unsigned int it=0; it< pixelNotToCoverX.size(); ++it ){
		pixelNotToCover[it] = pixelNotToCoverY[it]*width + pixelNotToCoverX[it];
	}

	return coveringCC(pixelToCover, pixelNotToCover);

}
/********************** attributes ***********************************/
void momentsAddPixel(ui x, ui y, vector<double>& att){
	int nbm=4;
	if(att.size()==0){
		att.resize(nbm*nbm);
		fill_n(begin(att),nbm*nbm,0.0);
	}
	double cx=1;
	double cy=1;

	for(int i=0; i<nbm; ++i){
		cy=1;
		for(int j=0; j<nbm; ++j){
			att[j*nbm+i]+=cx*cy;
			cy*=y;
		}
		cx*=x;
	}
}
void momentsMerge(vector<double>& paratt, vector<double>& att){
	for(ui i=0; i< paratt.size(); ++i){
		paratt[i]+=att[i];
	}
}
vector< vector<double> > centralMoments(const vector< vector<double> >& M){
	vector < vector < double > > mu(4, vector< double >(4,0) );
	double x_bar = M[1][0] / M[0][0];
	double y_bar = M[0][1] / M[0][0];
	mu[0][0] = M[0][0];
	mu[0][1] = 0;
	mu[1][0] = 0;
	mu[1][1] = M[1][1] - x_bar*M[0][1];
	mu[2][0] = M[2][0] - x_bar*M[1][0];
	mu[0][2] = M[0][2] - y_bar*M[0][1];
	mu[2][1] = M[2][1] - 2*x_bar*M[1][1] - y_bar*M[2][0] + 2* x_bar*x_bar*M[0][1];
	mu[1][2] = M[1][2] - 2 *y_bar*M[1][1] - x_bar*M[0][2] + 2* y_bar*y_bar*M[1][0];
	mu[3][0] = M[3][0] - 3 *x_bar*M[2][0] + 2 * x_bar*x_bar*M[1][0];
	mu[0][3] = M[0][3] - 3 * y_bar * M[0][2] + 2 * y_bar*y_bar* M[0][1];
	return mu;
}

vector < double > huMoments(const vector < double > &att){
	vector < vector < double > > M(4, vector< double >(4,0) );
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j)
			M[i][j]=att[j*4 + i];
	}

	vector < vector<double> > mu = centralMoments(M);
	vector < vector < double > > nabla(4, vector< double >(4,0) );
	for(int i=0;i<4;++i)
		for(int j=0;j<4;++j)
			nabla[i][j] = mu[i][j] / pow(mu[0][0], 1.0 + (i+j)*0.5);

	vector<double> hu(7);
	hu[0] = nabla[2][0]+nabla[0][2];
	hu[1] = pow( nabla[2][0]-nabla[0][2] , 2) + 4* pow(nabla[1][1],2);
	hu[2] = pow(nabla[3][0] - 3*nabla[1][2],2) + pow(3*nabla[2][1] - nabla[0][3],2);
    hu[3] = pow(nabla[3][0] + nabla[1][2], 2) + pow(nabla[2][1] + nabla[0][3],2);
    hu[4] = (nabla[3][0] - 3*nabla[1][2])*(nabla[3][0] + nabla[1][2])*(pow(nabla[3][0] + nabla[1][2],2) - 3*pow(nabla[2][1] + nabla[0][3], 2))
    		+ (3*nabla[2][1] - nabla[0][3])*(nabla[2][1] + nabla[0][3])*( 3*pow(nabla[3][0] + nabla[1][2],2) -  pow(nabla[2][1] + nabla[0][3],2));
    hu[5] =  (nabla[2][0] - nabla[0][2])*(pow(nabla[3][0] + nabla[1][2],2) - pow(nabla[2][1] + nabla[0][3],2)) +
    		4*nabla[1][1]*(nabla[3][0] + nabla[1][2])*(nabla[2][1] + nabla[0][3]);
    hu[6] = (3*nabla[2][1] - nabla[0][3])*(nabla[3][0] + nabla[1][2])*(pow(nabla[3][0] + nabla[1][2],2) - 3*pow(nabla[2][1] + nabla[0][3],2))
    		- (nabla[3][0] - 3*nabla[1][2])*(nabla[2][1] + nabla[0][3])*(3*pow(nabla[3][0] + nabla[1][2],2) - pow(nabla[2][1] + nabla[0][3],2));

    return hu;
}

vector < double > pcaMoments(const vector <double>& att){
	vector < vector < double > > M(4, vector< double >(4,0) );
	for(int i=0;i<4;++i)
		for(int j=0;j<4;++j)
			M[i][j]=att[j*4 + i];

	vector < vector<double> > mu = centralMoments(M);
	double mu_11 = mu[1][1]/mu[0][0];
	double mu_20 = mu[2][0]/mu[0][0];
	double mu_02 = mu[0][2]/mu[0][0];
	double angle = 0.5* atan2(2*mu_11, mu_20 - mu_02);
	double delta = 0.5* sqrt(4*pow(mu_11,2)+ pow(mu_20-mu_02,2));
	double pca_big = 0.5*(mu_20+mu_02) + delta;
	double pca_small = 0.5*(mu_20+mu_02) - delta;

	vector < double > pca_attr(3); // angle, pca_big, pca_small
	pca_attr[0] = angle;
	pca_attr[1] = pca_big;
	pca_attr[2] = pca_small;
    return pca_attr;
}

vector < double > shapeAttributes(const vector <double>& bbox_att, const vector <double>& mom_att){
	vector <double> natt(15); // xmin, ymin, xmax, ymax, area, angle, pca_big, pca_small, hu_1, hu_2, ... , hu_7

	natt[0] = bbox_att[0]; // xmin
	natt[1] = bbox_att[1]; // ymin
	natt[2] = bbox_att[2]; // xmax
	natt[3] = bbox_att[3]; // ymax

	natt[4] = mom_att[0]; // area
	vector <double> pca_attr = pcaMoments(mom_att);
	vector <double> hu = huMoments(mom_att);
	copy(begin(pca_attr),end(pca_attr),begin(natt)+5);
	copy(begin(hu),end(hu),begin(natt)+8);

	return natt;
}

void bboxAddPixel(ui x, ui y, vector<double>& att){
	if(att.size()==0){
		att.resize(4);
		fill_n(begin(att),4,0.0);
		att[0]=x; att[1]=y; // xmin, ymin
		att[2]=x; att[3]=y; // xmax, ymax
	}
	if (x<att[0])
		att[0]=x;
	if (x>att[2])
		att[2]=x;
	if (y<att[1])
		att[1]=y;
	if (y>att[3])
		att[3]=y;

}
void bboxMerge(vector<double>& paratt, vector<double>& att){
	if (att[0]<paratt[0])
		paratt[0]=att[0];
	if (att[2]>paratt[2])
		paratt[2]=att[2];
	if (att[1]<paratt[1])
		paratt[1]=att[1];
	if (att[3]>paratt[3])
		paratt[3]=att[3];
}

template < typename PixelType >
vector < vector < double > > MaxTree<PixelType>::computeShapeAttributes( ){
	// computes 15 shape attributes
	// xmin, ymin, xmax, ymax, area, angle, pca_big, pca_small, hu_1, hu_2, ... , hu_7

	vector < vector < double > > mom_att( getNbCC() );
	vector < vector < double > > bbox_att( getNbCC() );
	int cc_idx=0;
	for(ui p=0;p<nbpixels;++p){
		ui x = p%width;
		ui y = p/width;
		if(diff[p]){
		    cc_idx = pixelheader2cc[p];
		}else{
		    cc_idx = pixelheader2cc[ parent[p] ];
		}
		momentsAddPixel(x,y,mom_att[ cc_idx ]);
		bboxAddPixel(x,y,bbox_att[ cc_idx ]);
	}

	for(ui cc=0; cc<getNbCC();++cc){
		ui pixhead = cc2pixelheader[cc];
		ui parent_pixhead = parent[pixhead];
		if(pixhead!=parent_pixhead){
			ui parentcc = pixelheader2cc[parent_pixhead];
			momentsMerge(mom_att[parentcc], mom_att[cc] );
			bboxMerge(bbox_att[parentcc], bbox_att[cc] );
		}
	}

	vector < vector < double > > shp_att( getNbCC() );
	for(ui cc=0; cc<getNbCC(); ++cc ){
		vector <double> locatt = shapeAttributes(bbox_att[cc],mom_att[cc]);
		shp_att[cc].resize(locatt.size());
		copy(begin(locatt),end(locatt),begin(shp_att[cc]));
	}

	return shp_att;
}

void layerAddPixel(ui p, vector<double>& att, float layerVal){
	// sum_layer, sum_layer*layer, min_val, max_val
	if(att.size()==0){
		att.resize(5);
		fill_n(begin(att), 5 ,0.0);
		att[3] = layerVal;
		att[4] = layerVal;
	}
	att[0]+=1; // for number in sum
	att[1]+=layerVal; // for sum
	att[2]+=pow(layerVal,2); // for sum of square
	if(layerVal<att[3]) // for min
		att[3]=layerVal;
	if(layerVal>att[4]) // for max
		att[4]=layerVal;

}
void layerMerge(vector<double>& paratt, vector<double>& att){
	paratt[0]+=att[0];
	paratt[1]+=att[1];
	paratt[2]+=att[2];
	if(att[3]<paratt[3])
		paratt[3]=att[3];
	if(att[4]>paratt[4])
		paratt[4]=att[4];
}

vector < double > layerAttributes(const vector<double>&att){
	// create 4 layer attributes per CC: average, std, min_val, max_val
	vector <double> natt(4);

	natt[0] = att[1]/att[0]; // average
	natt[1] = sqrt( att[2]/att[0] - pow(natt[0],2)); // std
	natt[2] = att[3];
	natt[3] = att[4];

	return natt;
}
template < typename PixelType >
vector < vector < double > > MaxTree<PixelType>::computeLayerAttributes( const vector< float > &layer){
	// computes 4 layer attributes
	// features are: average, std, min_val, max_val
	vector < vector < double > > layer_att( getNbCC() );
	for(ui p=0;p<nbpixels;++p){
		if(diff[p]){
			layerAddPixel(p,layer_att[ pixelheader2cc[p] ], layer[p]);
		}else{
			layerAddPixel(p,layer_att[ pixelheader2cc[ parent[p] ]], layer[p]);
		}
	}

	for(ui cc=0; cc<getNbCC();++cc){
		ui pixhead = cc2pixelheader[cc];
		ui parent_pixhead = parent[pixhead];
		if(pixhead!=parent_pixhead){
			ui parentcc = pixelheader2cc[parent_pixhead];
			layerMerge(layer_att[parentcc], layer_att[ cc] );
		}
	}

	for(ui cc=getNbCC() ; cc>=1; --cc ){
		ui parentcc = pixelheader2cc[ parent[cc2pixelheader[ cc-1 ]] ];
		layerMerge(layer_att[parentcc], layer_att[ cc-1 ] );
	}

	vector < vector < double > > lyr_att(  getNbCC() );
	for(ui cc=0; cc<getNbCC(); ++cc){
		vector <double> locatt = layerAttributes(layer_att[cc]);
		lyr_att[cc].resize(locatt.size());
		copy(begin(locatt),end(locatt),begin(lyr_att[cc]));
	}
	return lyr_att;
}

template < typename PixelType >
void MaxTree<PixelType>::computeShapeAttributes_swig(float ** out, unsigned int* w, unsigned int *h){
	// for swig
	// do the computation
	vector < vector < double > > res = computeShapeAttributes();


	// copy the filtering into the output
	ui width = res.size();
	*w = width;
	ui height = res[0].size();
	*h = height;
	*out = new float[width*height];
	for(ui j=0;j< width;++j){
		for(ui i=0; i< height;++i)
			out[0][j*height+i] = res[j][i];
	}

}
template < typename PixelType >
void MaxTree<PixelType>::computeLayerAttributes_swig(float * imarray, unsigned int width, unsigned int height,
		float ** out, unsigned int* w, unsigned int *h){

	// for swig
	unsigned int s = width*height;
	vector< float > layer(s);
	for(ui i=0;i<s;++i)
		layer[i] = imarray[i];

	// do the computation
	vector < vector < double > > res = computeLayerAttributes(layer);

	// copy the filtering into the output
	ui widthh = res.size();
	*w = widthh;
	ui heighth = res[0].size();
	*h = heighth;
	*out = new float[widthh*heighth];
	for(ui j=0;j< widthh;++j){
		for(ui i=0; i< heighth;++i)
			out[0][j*heighth+i] = res[j][i];
	}

}
/*****************************************************************************/
template<typename PixelType>
inline vector< float > MaxTree<PixelType>::_filter_feature(
		ui curHeader,
		map< ui, vector<float> > & pixelheader2value,
		const map< ui, float> & isretained,
		const map< ui, vector<float> > & feature_values) {

	ui locparent = parent[curHeader];

	auto f = pixelheader2value.find(curHeader);

	if(f==end(pixelheader2value)){ // if the header is not in the map
		float dif =  (float) diff[curHeader];
		float value = isretained.find(curHeader)->second;
		vector < float > prev_values = feature_values.find(curHeader)->second;
		float count = prev_values[0] * dif ;
		float sum = prev_values[1] * dif ;
		float sumsquare = prev_values[2] * dif ;
		float minn = prev_values[3];
		float maxx = prev_values[4];

		vector < float > values(5);

		if(locparent!=curHeader){ //if the header is not the  root
			float pvalue = isretained.find(locparent)->second;
			vector < float > tmp_values = _filter_feature(locparent,pixelheader2value, isretained, feature_values);
			if (value>0.0){
				if(pvalue>0.0){
					values[0]=tmp_values[0]+count;
					values[1]=tmp_values[1]+sum;
					values[2]=tmp_values[2]+sumsquare;
					values[3]=min(tmp_values[3],minn);
					values[4]=max(tmp_values[4],maxx);
				}else{
					values[0]=count;
					values[1]=sum;
					values[2]=sumsquare;
					values[3]=minn;
					values[4]=maxx;
				}
			}else{
				copy(tmp_values.begin(), tmp_values.end(), values.begin());
			}
		}else{
			copy(prev_values.begin(), prev_values.end(), values.begin());
		}
		pixelheader2value[curHeader] = values;
		return values;

	}else{					// if the header is in the map
		return f->second;
	}
}

template<typename PixelType>
inline map <ui, vector< float > > MaxTree<PixelType>::_filterall_feature(
		const map< ui, float> & isretained,
		const map< ui, vector<float> > & feature_values){

	map < ui , vector<float> > pixelheader2value;
	for (auto pixelheader: pixelheader2cc){
		_filter_feature(pixelheader.first, pixelheader2value, isretained, feature_values);
	}
	return pixelheader2value;
}

template<typename PixelType>
inline void MaxTree<PixelType>::_filterallpixels_feature(
		vector < vector<float> > & res,
		const map< ui, float> & isretained,
		const map< ui, vector< float > > & feature_values){

	map < ui , vector< float > > pixelheader2values = _filterall_feature(isretained, feature_values);
	// map the values to count, avg, std, min ,max
	//for(auto pich_values : pixelheader2values){
	//	float avg = pich_values.second[1]/pich_values.second[0];
	//	float std = pich_values.second[2]/pich_values.second[0] - avg*avg;
	//	std = sqrt(std);
	//	pich_values.second[1] = avg;
	//	pich_values.second[2] = std;
	//}
	// copy the values to the multi dimensional image
	for(ui i=0; i<nbpixels;++i){
		vector < float > tmp_res;
		if(diff[i]){ // that a header pixel
					tmp_res=pixelheader2values[i];
		}else{
					tmp_res=pixelheader2values[parent[i]];
		}
		res[i].resize(5);
		copy(tmp_res.begin(),tmp_res.end(),res[i].begin());
		float avg = tmp_res[1]/(tmp_res[0]+0.0001);
		float std = tmp_res[2]/(tmp_res[0]+0.0001) - avg*avg;
		std = sqrt(std);
		res[i][1] = avg;
		res[i][2] = std;
		//cout << tmp_res[1]<< " " ;
	}
	return;
}
template < typename PixelType >
vector < vector<float> > MaxTree<PixelType>::computePerPixelAttributes(const vector < ui > & retained, const vector< float > &features){
	// retained is a vector of length = nb of CC to remap
	vector < vector<float> > res(nbpixels);

	// populate is retained vector
	map< ui, float> isretained;
	map< ui, vector< float > > feature_values;
	for(auto header : pixelheader2cc){
		isretained[header.first] = 0.0;
		feature_values[header.first].resize(5);
	}
	for(ui cc=0; cc < retained.size(); ++cc){
		if ( retained[cc] < cc2pixelheader.size() ){
			ui ph = cc2pixelheader[ retained[cc] ] ;
			isretained[ ph ] = 1.0; // indicate if components is kept
			feature_values[ ph ][0] = 1; //count
			feature_values[ ph ][1] = features[cc]; //sum
			feature_values[ ph ][2] = features[cc]*features[cc]; //sum of square
			feature_values[ ph ][3] = features[cc]; // min
			feature_values[ ph ][4] = features[cc]; // max
		}
	}

	//launch the procedure
	_filterallpixels_feature(res, isretained, feature_values);
	return res;
}
template < typename PixelType >
void MaxTree<PixelType>::computePerPixelAttributes_swig(float ** outf, unsigned int *wf, unsigned int *hf,
		 unsigned int* retained, unsigned int lr, float* score, unsigned int ls){
	// copy the filtering into the output
	int n_height = height*5;
	*wf = n_height;
	*hf = width;
	*outf = new float[nbpixels*5];

	//copy retained and score
	vector < ui > retained_v(lr);
	vector < float > score_v(ls);
	copy(retained, retained+lr, retained_v.begin());
	copy(score, score+ls, score_v.begin());

	// perform computation
	vector < vector< float > > per_pixel_attributes = computePerPixelAttributes(retained_v, score_v);
	for(ui j=0;j< width;++j){
			for(ui i=0; i< height;++i){
				for(ui u=0; u< 5; ++u){
				outf[0][(j*height+i)+u*nbpixels] = per_pixel_attributes[j*height+i][u];
				}
			}
	}
}

/********************** Templates ************************************/
template class MaxTree<unsigned char>;
template class MaxTree<char>;
template class MaxTree<unsigned short>;
template class MaxTree<short>;
template class MaxTree<int>;
template class MaxTree<unsigned int>;
template class MaxTree<float>;
