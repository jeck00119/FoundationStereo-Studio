"""The two side panels.

A package rather than one 1749-line module: InputPanel and ParamPanel are
independent classes that only shared a file. Everything the rest of the app (and
the tests) imported from ``studio.panels`` is re-exported here, so this split
changed no import anywhere.
"""
from ._common import (_BASELINE_CFG, _BOX_DEFAULT_MM, _BOXC_CFG, _BOXS_CFG,
                      _REF_TIP, _ZFAR_CFG, _ZNEAR_CFG, build_param_widgets,
                      field_row, make_spin, np_to_qpixmap, read_param_widgets,
                      sanitize_params, set_param_widgets)
from .input import InputPanel
from .params import ParamPanel

__all__ = ["InputPanel", "ParamPanel", "np_to_qpixmap", "make_spin", "field_row",
           "build_param_widgets", "read_param_widgets", "set_param_widgets",
           "sanitize_params"]
