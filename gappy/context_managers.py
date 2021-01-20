"""
Context Managers for gappy.

This module implements a context manager for global variables. This is
useful since the behavior of GAP is sometimes controlled by global
variables, which you might want to switch to a different value for a
computation. Here is an example how you are suppose to use it from
your code. First, let us set a dummy global variable for our example::

    >>> gap.set_global('FooBar', 123)

Then, if you want to switch the value momentarily you can write::

    >>> with gap.global_context('FooBar', 'test'):
    ...     print(gap.get_global('FooBar'))
    test

Afterward, the global variable reverts to the previous value::

    >>> print(gap.get_global('FooBar'))
    123

The value is reset even if exceptions occur::

    >>> with gap.global_context('FooBar', 'test'):
    ...     print(gap.get_global('FooBar'))
    ...     raise ValueError(gap.get_global('FooBar'))
    Traceback (most recent call last):
    ...
    ValueError: test
    >>> print(gap.get_global('FooBar'))
    123
"""


###############################################################################
#       Copyright (C) 2012, Volker Braun <vbraun.name@gmail.com>
#       Copyright (C) 2021, E. Madison Bray <embray@lri.fr>
#
#   Distributed under the terms of the GNU General Public License (GPL)
#   as published by the Free Software Foundation; either version 2 of
#   the License, or (at your option) any later version.
#                   http://www.gnu.org/licenses/
###############################################################################


__all__ = ['GlobalVariableContext']


class GlobalVariableContext:
    """
    Context manager for GAP global variables.

    It is recommended that you use the :meth:`~gappy.core.Gap.global_context`
    method and not construct objects of this class manually.

    Parameters
    ----------
    variable : str
        The GAP variable name.
    value
        Anything that defines or can be converted to a GAP object.

    Examples
    --------

    >>> gap.set_global('FooBar', 1)
    >>> with gap.global_context('FooBar', 2):
    ...     print(gap.get_global('FooBar'))
    2
    >>> gap.get_global('FooBar')
    1
    """

    def __init__(self, gap, variable, value):
        self._gap = gap
        self._variable = variable
        self._new_value = value

    def __enter__(self):
        """
        Called when entering the with-block

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> with gap.global_context('FooBar', 2):
        ...     print(gap.get_global('FooBar'))
        2
        >>> gap.get_global('FooBar')
        1
        """
        self._old_value = self._gap.get_global(self._variable)
        self._gap.set_global(self._variable, self._new_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting the with-block

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> with gap.global_context('FooBar', 2):
        ...     print(gap.get_global('FooBar'))
        2
        >>> gap.get_global('FooBar')
        1
        """
        self._gap.set_global(self._variable, self._old_value)
        return False
