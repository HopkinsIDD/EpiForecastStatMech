"""
### IntensityFamily, an abstraction to enable work across mechanistic models.
"""
import collections

class IntensityFamily(
    collections.namedtuple('IntensityFamily', [
        'name', 'intensity', 'params_wrapper', 'params0', 'param_names',
        'encoded_param_names'
    ])):
  pass
