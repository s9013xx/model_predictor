from .perfnetA import *
from .resperfnet import *
import os
from pkgutil import iter_modules

__all__ = []

def _global_import(name, all_list):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else []
    if lst:
        del globals()[name]
        for k in lst:
            globals()[k] = p.__dict__[k]
            all_list.append(k)

_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules([_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    _global_import(module_name, __all__)

def get_local_nn(inputs, training, network_name="perfnetA", pred = None):
    nn_map = {
        'perfnetA': perfnetA(inputs, training),
        'resperfnet': resperfnet(inputs, training),
    }
    return nn_map[network_name]

def get_nn_list():
    return _global_import(module_name, [])

__all__.append('get_local_nn')
#print(__all__)
