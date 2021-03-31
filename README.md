# maxtree
Implements the Maxtree algorithm with python binding, using numpy or torch. 
It extends the original algorithm to a [Differentiable Maxtree](publication/README.md) layer which can be integrated
in a deep learning architecture.

## Installation
### pip
```shell script
pip install maxtree
```

### dev
```shell script
git clone https://github.com/gueguenster/maxtree && cd maxtree
# conda install --channel https://conda.anaconda.org/anaconda clangxx_osx-64 for macos
pip install -e .
```

## Usage example
### Differential maxtree - pytorch
Get theoritical description at: [Differentiable Maxtree](publication/README.md)

See examples at: [example/differential.ipynb](example/differential.ipynb)
```python
import torch
from maxtree.maxtree_torch import DifferentialMaxtree

batch, num_channels, h, w = 1, 10, 32, 64
layer = DifferentialMaxtree(num_channels=num_channels)

input = torch.randint(-10, 32, (batch, num_channels, h, w))  # the input is casted to torch.int16, so inputs range
                                                      # must be adequately scaled to loose information.
                                                      # the larger the range is, the longer the computation.
output = layer(input)  # output is the result of Differential Maxtree filtering, of size  (batch, num_channels, h, w)
                       # output is floating point.
```

### Numpy
see [example/example.py](example/example.py)
```python
import matplotlib.pyplot as plt
import numpy as np
from maxtree.component_tree import MaxTree
from scipy import misc

img_rgb = misc.face()
img = np.uint16(img_rgb[:,:,0])

# Example 1
mt = MaxTree(img) # compute the max tree
mt.compute_shape_attributes() # compute shape attributes
cc_areas = mt.getAttributes('area') # retrieve the area of each connected components

idx_retained = np.logical_and(cc_areas>800, cc_areas<5000).nonzero()[0] # select the cc with an 800<area<5000
filtered_out = mt.filter(idx_retained) # direct filtering of the cc


plt.figure()
plt.subplot(1,3,1)
plt.imshow(img)
plt.title('first of rgb image')

plt.subplot(1,3,2)
plt.imshow(filtered_out)
plt.title('cc having an area between 800 and 5000')

ax = plt.subplot(1,3,3)
plt.hist(cc_areas, bins=12)
plt.xlabel('area in nb of pixels')
ax.set_yscale("log", nonposy='clip')
plt.show()

# Example 2
img_c = img.max() - img
mt = MaxTree(img_c) # compute max tree

lyr = np.float32(img_rgb[:,:,1]) # compute a layer
mt.compute_layer_attributes(lyr) # compute layer features per cc
print mt 

avg_std = mt.getAttributes(['average_0','std_0']) # retrieve the layer average and std per cc
idx_retained = np.logical_and(avg_std[:,0]<100, avg_std[:,1]<30).nonzero()[0] # select the cc 
filtered_out = mt.filter(idx_retained) # direct filtering

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_c)
plt.title('complement of image')

plt.subplot(1,2,2)
plt.imshow(filtered_out)
plt.title('cc having an average layer < 50 and std layer < 10')
plt.show()

# Example 3
mt.compute_shape_attributes()
cc_xmin = mt.getAttributes('xmin')
cc_xmax = mt.getAttributes('xmax')
scores = np.exp(-np.abs(cc_xmin-350)/100 -np.abs(cc_xmax-450)/100)

idx_retained = np.uint32(np.arange(mt.nb_connected_components))

filtered_out = mt.filter(idx_retained,scores)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_c)
plt.title('complement of image')

plt.subplot(1,2,2)
plt.imshow(filtered_out)
plt.title('cc with scores')
plt.show()
```
