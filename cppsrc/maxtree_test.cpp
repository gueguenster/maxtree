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


#include <limits.h>
#include "gtest/gtest.h"

#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();

}


#include "maxtree.h"
class MaxTreeTest : public ::testing::Test {
 public:

	MaxTree <unsigned short> mtus;
	MaxTree <short> mts;
	MaxTree <float > mtf;
	MaxTree <unsigned char > mtuc;

  virtual void SetUp() {
	  unsigned int width=15;
	  unsigned int height=5;
	  unsigned int nbp = width*height;

	  srand (time(NULL));

	  vector<unsigned short> imus(nbp);
	  vector<short> ims(nbp);
	  vector<float> imf(nbp);



	  for(int i=0;i<nbp;++i){
		  imus[i] = rand()%100;
		  ims[i] = rand()%100;
		  imf[i] = ((float)(rand()%100)) / 100;
	  }

	  mtus.readim(imus,width,height);
	  mts.readim(ims,width,height);
	  mtf.readim(imf,width,height);

	  unsigned int ww = 10;
	  unsigned int hh = 10;
	  vector < unsigned char > imuc(ww*hh);
	  for (int i=0; i < ww*hh; i++){
		  int x = i%ww;
		  int y = i/ww;
		  if(x>=3 && x<=7 && y>=3 && y<=7){
			  imuc[i] = 8;
		  }else{
			  imuc[i]=1;
		  }
	  }
	  imuc[5*ww+5] = 9;

	  mtuc.readim(imuc,ww,hh);


  }


  virtual void TearDown() {

  }
};


TEST_F(MaxTreeTest, ReadImage) {
	EXPECT_EQ(0,0);
}


TEST_F(MaxTreeTest, ComputeMaxTreeRandom ) {
	mtus.compute();
	mtus._print();

	mts.compute();
	mts._print();

	mtf.compute();
	mtf._print();

	ASSERT_TRUE( true );
}

TEST_F(MaxTreeTest, ComputeMaxSimpleSquare ) {
	mtuc.compute();
	mtuc._print();

	ASSERT_TRUE( true );
}

TEST_F(MaxTreeTest, Connectivity ) {
	MaxTree<unsigned char> r_mts(mtuc.getIm(), mtuc.getWidth(), mtuc.getHeight(),8);
	r_mts.compute();
	r_mts._print();

	ASSERT_TRUE( r_mts.getConnectivity()==8 );

	MaxTree<unsigned short> r_mt(mtus.getIm(), mtus.getWidth(), mtus.getHeight(),8);
	r_mt.compute();
	r_mt._print();

	ASSERT_TRUE( r_mt.getConnectivity()==8 );
}

TEST_F(MaxTreeTest, RecomputeImage ) {
	 ui w=100; ui h=150;
	 vector<unsigned short> ims(w*h);

	 for(int i=0;i<w*h;++i){
		  ims[i] = rand()%100;
	 }
	 MaxTree<unsigned short> mts(ims,w,h);
	 mts.compute();

	 MaxTree<unsigned short> r_mts(mts.getParent(), mts.getDiff(), w, h);
	 const vector<unsigned short> rim = r_mts.getIm();
	 for(int i=0;i<w*h;++i){
		 ASSERT_TRUE( rim[i]==ims[i] );
	 }

}

TEST_F(MaxTreeTest, FilterImage ) {
	mtuc.compute();
	//mtuc._print();

	vector < ui > r(1); r[0] = 1;
	vector < unsigned char > v = mtuc.filter(r);
	for(ui j=0; j< mtuc.getHeight(); ++j){
		for(ui i=0; i <mtuc.getWidth(); ++i){
			cout << (int)v[j*mtuc.getWidth()+i] << " ";
			ASSERT_TRUE(v[j*mtuc.getWidth()+i]>=0 && v[j*mtuc.getWidth()+i]<=7 );
		}
		cout << endl;

	}

	vector < pair< ui,float > > rs(1); rs[0].first = 1; rs[0].second=0.5;
	ASSERT_TRUE(mtuc.getNbCC()==3);
	vector < float > vs = mtuc.filter(rs);
	float ma= *max_element(vs.begin(), vs.end() );
	cout << "maximum fuzzy :" << ma << endl;
	ASSERT_TRUE(ma==3.5);

}

TEST_F(MaxTreeTest, FilterFuzzyImage ) {

	mtus.compute();

	vector < pair< ui, float > > rs(mtus.getNbCC());
	ui count=0;
	for(auto& a : rs){
		a.first = count;
		count++;
		a.second = (float)(rand()%10) /10.0;

	}

	vector < float > vs = mtus.filter(rs);

	for(ui j=0; j< mtus.getHeight(); ++j){
		for(ui i=0; i <mtus.getWidth(); ++i){
			cout << "(" << vs[j*mtuc.getWidth()+i] << ", " << mtus.getIm()[j*mtuc.getWidth()+i]<< ") ";
			ASSERT_TRUE(vs[j*mtuc.getWidth()+i] <=  mtus.getIm()[j*mtuc.getWidth()+i] );
		}
		cout << endl;

	}
}

TEST_F(MaxTreeTest, CoveringCC ) {
	mtuc.compute();

	vector < ui > tocover(1); tocover[0] = 5*mtuc.getWidth()+5;
	vector < ui > notocover(1); notocover[0] = 0*mtuc.getWidth()+0;

	vector < ui > r = mtuc.coveringCC(tocover,notocover);

	vector < unsigned char > v = mtuc.filter(r);
	for(ui j=0; j< mtuc.getHeight(); ++j){
		for(ui i=0; i <mtuc.getWidth(); ++i){
			cout << (int)v[j*mtuc.getWidth()+i] << " ";

		}
		cout << endl;

	}
	ASSERT_TRUE(r.size()==2);


	vector < ui > rs = mtuc.coveringCC(tocover);

		vector < unsigned char > vs = mtuc.filter(rs);
		for(ui j=0; j< mtuc.getHeight(); ++j){
			for(ui i=0; i <mtuc.getWidth(); ++i){
				cout << (int)vs[j*mtuc.getWidth()+i] << " ";

			}
			cout << endl;

		}
		ASSERT_TRUE(rs.size()==3);
}

TEST_F(MaxTreeTest, Moments ) {
	mtuc.compute();
	vector < vector < double> > att = mtuc.computeShapeAttributes();
	for(auto v : att){
		for(auto it:v )
			cout << it << " ";
		cout << endl;
	}

}

TEST_F(MaxTreeTest, Layers ) {
	mtuc.compute();
	const vector<unsigned char> rim = mtuc.getIm();
	vector<float> layer(rim.size());
	for(int i=0;i<rim.size();++i)
		layer[i] = rim[i];
	vector<vector<double> > res = mtuc.computeShapeAttributes();
	cout<< "moments features" << endl;
	cout<< "xmin, ymin, xmax, ymax, area, angle, pca_big, pca_small, hu_1, hu_2, ... , hu_7" << endl;
	for(auto v : res){
		for(auto it:v )
			cout << it << " ";
		cout << endl;
	}
	ASSERT_TRUE(res[0][0]==5);
	ASSERT_TRUE(res[2][4]==100);
	cout<< "moments features" << endl;
	cout<< "features are: average, std, min_val, max_val" << endl;
	vector < vector < double> > att = mtuc.computeLayerAttributes(layer);
	for(auto v : att){
		for(auto it:v )
			cout << it << " ";
		cout << endl;
	}
	ASSERT_TRUE(att[0][1]==0.0);
	ASSERT_TRUE(att[1][2]==layer[35]);

}

TEST_F(MaxTreeTest, Scalability ) {
	  unsigned int width=200;
	  unsigned int height=223;
	  unsigned int nbp = width*height;

	  vector<unsigned short> imus(nbp);
	  srand(45);
	  for(int i=0;i<nbp;++i){
		  imus[i] = rand()%100;
	  }
	  MaxTree <unsigned short> locmtus;
	  locmtus.readim(imus,width,height);
	  locmtus.compute();
	  vector<vector<double> > res = locmtus.computeShapeAttributes();
	  //cout << res.size() << endl;
	  cout << "size moments " << res[0].size() << endl;
	  ASSERT_TRUE(res[0].size()==15);

	  vector < double > lastMoment = *(end(res)-1);
	  for(auto it:lastMoment )
		  cout << it << " ";
	  cout << endl;
	  ASSERT_TRUE(lastMoment[4]==nbp);
	  ASSERT_TRUE(lastMoment[0]==0);
	  ASSERT_TRUE(lastMoment[2]==width-1);
	  ASSERT_TRUE(lastMoment[3]==height-1);

	 cout << "compute layer attributes "<< endl;
	 vector<float> layer(nbp);
	 for(int i=0;i<nbp;++i)
		layer[i] = rand()%100;
	  vector < vector < double> > att = locmtus.computeLayerAttributes(layer);
	  ASSERT_TRUE(res.size() == locmtus.getNbCC());
	  for(auto it:att[0] )
		  cout << it << " ";
}



