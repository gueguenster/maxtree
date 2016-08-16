# maxtree
implements the maxtree algorithm with python binding

# requirements
* install google test: https://github.com/google/googletest
* install numpy >=1.8
* install g++ supporting the -std=c++11 option
* install swig: https://github.com/swig/swig

# building from source
```
 git clone https://github.com/gueguenster/maxtree
 cd maxtree/maxtree
 python setup.py build && python setup.py install --force
 python -m unittest component_tree_test
```
