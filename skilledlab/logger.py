from typing import Union, List, Tuple, overload, Dict
from skilledlab.internal.util.colors import StyleCode

class Style(StyleCode):
    r"""
    Output styles
    """

    none = None
    normal = 'normal'
    bold = 'bold'
    underline = 'underline'
    light = 'light'


class Color(StyleCode):
    r"""
    Output colors
    """

    none = None
    black = 'black'
    red = 'red'
    green = 'green'
    orange = 'orange'
    blue = 'blue'
    purple = 'purple'
    cyan = 'cyan'
    white = 'white'


class Text(StyleCode):
    r"""
    Standard styles we use in labml
    """

    none = None
    danger = Color.red.value
    success = Color.green.value
    warning = Color.orange.value
    meta = Color.blue.value
    key = Color.cyan.value
    meta2 = Color.purple.value
    title = [Style.bold.value, Style.underline.value]
    heading = Style.underline.value
    value = Style.bold.value
    highlight = [Style.bold.value, Color.orange.value]
    subtle = [Style.light.value, Color.white.value]
    link = 'link'



