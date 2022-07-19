__all__ = ['DatasetDefinitionBase', 'get_global_definition', 'set_global_definition', 'gdef']


class DatasetDefinitionBase(object):
    translation_mean = None
    translation_std = None


class GlobalDefinitionWrapper(object):
    def __getattr__(self, item):
        return getattr(get_global_definition(), item)

    def __setattr__(self, key, value):
        raise AttributeError('Cannot set the attr of `gdef`.')


gdef = GlobalDefinitionWrapper()

_GLOBAL_DEF = None


def get_global_definition():
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is not None
    return _GLOBAL_DEF


def set_global_definition(def_):
    global _GLOBAL_DEF
    # assert _GLOBAL_DEF is None, print('GLOBAL_DEF:', _GLOBAL_DEF)
    _GLOBAL_DEF = def_
