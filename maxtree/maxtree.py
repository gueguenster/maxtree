# swig -python -c++ -I../cppsrc -o maxtree_wrap.cpp maxtree.i
# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_maxtree')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_maxtree')
    _maxtree = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_maxtree', [dirname(__file__)])
        except ImportError:
            import _maxtree
            return _maxtree
        try:
            _mod = imp.load_module('_maxtree', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _maxtree = swig_import_helper()
    del swig_import_helper
else:
    import _maxtree
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class MT(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MT, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MT, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _maxtree.new_MT(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def readim(self, imarray, width, height):
        return _maxtree.MT_readim(self, imarray, width, height)

    def compute(self):
        return _maxtree.MT_compute(self)

    def filter(self, *args):
        return _maxtree.MT_filter(self, *args)

    def filter_swig(self, *args):
        return _maxtree.MT_filter_swig(self, *args)

    def coveringCC(self, *args):
        return _maxtree.MT_coveringCC(self, *args)

    def coveringCC_XY(self, *args):
        return _maxtree.MT_coveringCC_XY(self, *args)

    def computeShapeAttributes(self):
        return _maxtree.MT_computeShapeAttributes(self)

    def computeLayerAttributes(self, layer):
        return _maxtree.MT_computeLayerAttributes(self, layer)

    def computeShapeAttributes_swig(self):
        return _maxtree.MT_computeShapeAttributes_swig(self)

    def computeLayerAttributes_swig(self, imarray):
        return _maxtree.MT_computeLayerAttributes_swig(self, imarray)

    def computePerPixelAttributes(self, retained, features):
        return _maxtree.MT_computePerPixelAttributes(self, retained, features)

    def computePerPixelAttributes_swig(self, retained, score):
        return _maxtree.MT_computePerPixelAttributes_swig(self, retained, score)

    def _print(self):
        return _maxtree.MT__print(self)

    def getConnectivity(self):
        return _maxtree.MT_getConnectivity(self)

    def setConnectivity(self, connectivity):
        return _maxtree.MT_setConnectivity(self, connectivity)

    def getHeight(self):
        return _maxtree.MT_getHeight(self)

    def getIm(self):
        return _maxtree.MT_getIm(self)

    def getNbpixels(self):
        return _maxtree.MT_getNbpixels(self)

    def getWidth(self):
        return _maxtree.MT_getWidth(self)

    def serialize_swig(self):
        return _maxtree.MT_serialize_swig(self)

    def getParent(self, *args):
        return _maxtree.MT_getParent(self, *args)

    def getDiff(self, *args):
        return _maxtree.MT_getDiff(self, *args)

    def getNbCC(self):
        return _maxtree.MT_getNbCC(self)

    def __getstate__(self):
        args={}
        args['bytes'] = self.serialize_swig()
        return args
    def __setstate__(self, state):
        dummy = int(0)
        self.__init__(dummy, state['bytes'])

    __swig_destroy__ = _maxtree.delete_MT
    __del__ = lambda self: None
MT_swigregister = _maxtree.MT_swigregister
MT_swigregister(MT)

# This file is compatible with both classic and new-style classes.


