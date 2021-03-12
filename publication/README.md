#Differential Maxtree
L. Gueguen, 2021

##Introduction

##Maxtree decomposition and filtering
###Maxtree decomposition
Maxtree are image (or signal) representations created by structuring the connected components resulting from threshold 
decomposition [1], [2]. They are also known as Component Trees [3]. They provide a multiscale description of extremas of
images and signals and are, among other things, one of the classical ways to build connected operators [4]. Let's define
few notations, to help describe the Maxtree decomposition. An image is a function mapping a pixel from grid G of dimension 
H x W to a value in the set of integers:

![equation](http://latex.codecogs.com/svg.latex?\mathbb{G}=&space;\[0\cdots&space;H-1\]&space;\times&space;\[0&space;\cdots&space;W-1])
![equation](http://latex.codecogs.com/svg.latex?I:&space;\[0\cdots&space;H-1\]&space;\times&space;\[0&space;\cdots&space;W-1]&space;\mapsto&space;\mathbb{N})

The set of all images adhering to the previous definition is called ![equation](http://latex.codecogs.com/svg.latex?\mathbb{I}).
A pixel from the grid is simply named ![equation](http://latex.codecogs.com/svg.latex?p&space;\in&space;\mathbb{G}), and its
image value is ![equation](http://latex.codecogs.com/svg.latex?I(p)&space;\in&space;\mathbb{N}&space;).

A threshold is an operator acting on an image, which indicates if a pixel is above a value or not:

![equation](http://latex.codecogs.com/svg.latex?T_%5Clambda%20(I(p))%20=%20%5Cbegin%7Bcases%7D%200,%20&%20I(p)%5Cleqslant%20%5Clambda%20%5C%5C1,%20&%20I(p)%20%3E%20%5Clambda%20%5Cend%7Bcases%7D)

And this threshold image can be decomposed in its connected components, given a connectivity criterion on the grid. The connected
components are subsets from ![equation](http://latex.codecogs.com/svg.latex?\mathbb{G) which are connected. We represent this collection
of connected components at some threshold value by:

![equation](http://latex.codecogs.com/svg.latex?C_%5Clambda(I)%20=%20%5C%7Bc%20%7C%20%5C:%20p%20%5Cin%20%5Cmathbb%7BG%7D,%20p%20%5Cin%20c,%20T_%5Clambda(I(p))=1,%20c%20%5C:%20%5Ctext%7Bis%20connected%7D%20%5C%7D)

The number of connected components is dependent on the input image. A connected component is a set of pixel positions on the grid.
A Maxtree extracts and represent efficiently all the connected components resulting from thresholding the image at all possible
thresholds between its minimum and maximum. The set of all connected components (CC) is:

![equation](http://latex.codecogs.com/svg.latex?C%5E&plus;(I)%20=%20%5Cbigcup_%7B%5Cmin(I)%20%5Cleq%20%5Clambda%20%5Cleq%20%5Cmax(I)%7D%7BC_%5Clambda(I)%7D%20)

This resulting set of CC has redundances which is eliminated by the union operator. The Maxtree algorithm efficientlty generates
the set of unique CCs, without the need to threshold the image at each threshold. In addition, it maintains a count of appearance for each 
CC in all the threshold sets, which corresponds to the number of thresholds containing the same CC. This histogram CCs is:

![equation](http://latex.codecogs.com/svg.latex?C(I)=%20%5C%7B(c,h)%20%7C%20c%20%5Cin%20C%5E&plus;(I),%20h%5Cin%20%5Cmathbb%7BN%7D%5E&plus;%20%5C%7D%20)

This CC histogram is a lossless representation of the image, as it allows to reconstruct the image. The function generating 
this CC histogram, as it is unique given an image, and it allows to generate back the input image:

![equation](http://latex.codecogs.com/svg.latex?I(p)%20=%20%5Csum_%7Bp%5Cin%20c,%20%5C;%20(c,%20h)%20%5Cin%20C(I)%7D%7Bh%7D%20)

The Maxtree generates the CC histogram by exploiting the nested nature of the CCs at different thresholds, and organizes 
them in a tree structure as illustrated below:

###Shape attributes
###Maxtree filtering

##Differential Maxtree filtering
###Maxtree differential filtering
###Backpropagation derivatives
###Torch based implementation

##Illustrations

##Conclusion

##References
[1] P. Salembier, A. Oliveras, and L. Garrido. Motion connected operators for image sequences, In VIII European Signal 
Processing Conference, EUSIPCO'96, pages 1083-1086, Trieste, Italy, September 1996.

[2] P. Salembier, A. Oliveras, and L. Garrido. Anti-extensive connected operators for image and sequence processing. 
IEEE Transactions on Image Processing, 7(4):555{570, April 1998.

[3] R. Jones. Component trees for image Filtering and segmentation. In 1997 IEEE Workshop on Nonlinear Signal and Image 
Processing, Mackinac Island, USA, 1997.

[4] P. Salembier and M. H. F. Wilkinson, Connected operators: A review of region-based morphological image processing 
techniques, IEEE Signal Processing Magazine, vol. 6, pp. 136–157, 2009.