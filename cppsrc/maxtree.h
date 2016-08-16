/*
COPYRIGHT

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

#ifndef MAXTREE_H_
#define MAXTREE_H_


#include <map>
#include <queue>
#include <vector>
#include <set>

using namespace std;

typedef unsigned int ui;


template <typename PixelType >
class MaxTree {
private:
	// some external structure for computation
	map< PixelType, queue< ui > > hierarchicalqueues; // the hierarchical queues

	// the tree structure data
	vector < ui > parent;		// the parent structure
	vector < PixelType > diff;  // the differences between head and parent
	ui width, height, nbpixels;

	//the feature structure data
	map < ui, ui > pixelheader2cc;		//from the position of a pixel map the cc it correspond to
	vector < ui > cc2pixelheader;	// from the CC, indicates its header

	//the image array
	vector < PixelType > im;    //the array containing the pixels

	//access method on the queues
	ui HQ_first(PixelType h); //return the first element of the queue at level h, and remove it
	void HQ_add(PixelType h, ui p); // add the position p at the end of the queue at level h
	bool HQ_noempty(PixelType h); // detrmine if the queue is not empty

	//neighborhood method
	vector < ui > GetNeighbors(ui p); // get the neighbors of p
	int connectivity; // either 4 or 8

	//compute function
	PixelType _compute(vector < bool > & hasBeenVisited, map< PixelType, ui > & leaders, PixelType h); // recursive function which floods each regions
	PixelType _computeImage(ui currentHeader, map <ui,PixelType> & pixelheader2value ); // recursive function used
	void computeImage(); // recompute the image
	void _resetMaps2(ui curHeader); // recursive function to reset the pixel2header, etc..

	//filtering methods
	inline float _filter( ui curHeader, map< ui, float > & pixelheader2value, const map< ui, float> & isretained);
	inline map <ui, float > _filterall(const map< ui, float> & isretained);
	inline void _filterallpixels(vector < float > & res, const map< ui, float> & isretained);

	//covering methods
	void _addcover(ui p, set < ui > & covering);

public:
	MaxTree();
	MaxTree(const vector<PixelType> & imarray, unsigned int width, unsigned int height );
	MaxTree(const vector<PixelType> & imarray, unsigned int width, unsigned int height , int connectivity);
	MaxTree(const vector<ui> & parent, const vector<PixelType> & diff, unsigned int width, unsigned int height);
	MaxTree(PixelType * imarray, unsigned int width, unsigned int height); // for swig
	MaxTree(unsigned int dummy, unsigned int * retained, unsigned int lr); // for swig pickling

	//io
	void readim(const vector<PixelType> & imarray, unsigned int width, unsigned int height);

	// compute the tree
	void compute();

	//filtering functions
	vector <PixelType> filter( const vector < ui > & retained );	// retained is a vector of the CC indices to remap
	vector <PixelType> filter( const vector < bool > & retained );  // retained is a vector of length = nb of CC to remap
	vector <float>     filter( const vector < pair< ui, float> > & retained ); // retained is a vector of CC indices with score to remap
	void filter_swig( unsigned int * retained, unsigned int lr,
			float* score, unsigned int ls,
			float ** outf, unsigned int* wf, unsigned int *hf); // for swig
	void filter_swig( unsigned int* retained, unsigned int lr,
			PixelType ** out, unsigned int* w, unsigned int *h); // for swig

	//covering functions
	vector < ui > coveringCC( const vector < ui > & pixelToCover); // returns the CC covering the given pixels
	vector < ui > coveringCC( const vector < ui > & pixelToCover, const vector < ui >& pixelNotToCover); // returns the CC covering the given pixels, but not the others
	vector < ui > coveringCC_XY( const vector < ui > & pixelToCoverX, const vector < ui > & pixelToCoverY); // returns the CC covering the given pixels
	vector < ui > coveringCC_XY( const vector < ui > & pixelToCoverX, const vector < ui > & pixelToCoverY,
							  const vector < ui > & pixelNotToCoverX, const vector < ui > & pixelNotToCoverY); // returns the CC covering the given pixels, but not the others

	// features computations
	vector < vector < double > > computeShapeAttributes( ); // computes 15 shape attributes per CC
	vector < vector < double > > computeLayerAttributes( const vector< float > &layer); // computes layer statistics per CC
	void computeShapeAttributes_swig(float ** outf, unsigned int *wf, unsigned int *hf); // for swig
	void computeLayerAttributes_swig(float * imarray, unsigned int width, unsigned int height,
			float ** outf, unsigned int *wf, unsigned int *hf);

	// print for small trees
	void _print(); // to b used for small images, use for test and debug

	//getters and setters
	int getConnectivity() const {
		return connectivity;
	}

	void setConnectivity(int connectivity) {
		if((connectivity==4)||(connectivity==8)){
			this->connectivity = connectivity;
		}else{
			this->connectivity = 4;
		}
	}

	ui getHeight() const {
		return height;
	}

	const vector<PixelType>& getIm() {
		computeImage();
		return im;
	}

	ui getNbpixels() const {
		return nbpixels;
	}

	ui getWidth() const {
		return width;
	}

	const vector<PixelType>& getDiff() const {
		return diff;
	}

	const vector<ui>& getParent() const {
		return parent;
	}

	void serialize_swig(unsigned int ** par, unsigned int *l) const {
		l[0] = nbpixels*2+2;
		par[0] = new unsigned int[l[0]];
		par[0][0] = width;
		par[0][1] = height;
		for(ui i=0; i< nbpixels; ++i){
			par[0][2*i+2] = parent[i];
			par[0][2*i+1+2] = diff[i];
		}
	}

	ui getParent(ui index){
		if (index < getNbCC()){
			ui p = parent[cc2pixelheader[index]];
			return pixelheader2cc[ p ];
		}else{
			return 0;
		}
	}

	float getDiff(ui index){
		if (index < getNbCC()){
			float d = diff[cc2pixelheader[index]];
			return d;
		}else{
			return 0;
		}
	}

	const ui getNbCC () const {
		return cc2pixelheader.size();
	}
};

#endif /* MAXTREE_H_ */
