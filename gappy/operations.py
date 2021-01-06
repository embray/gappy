"""
Operations for gappy objects.

GAP functions for which several methods can be available are called operations,
so GAP ``Size`` is an example of an operation. This module is for inspecting
GAP operations from Python. In particular, it can list the operations that take
a particular gappy object as its first argument.  This is used in tab
completion, where Python ``x.[TAB]`` lists all GAP operations for which
``Operation(x, ...)`` is defined.
"""

import re
import string


NAME_RE = re.compile(r'(Setter|Getter|Tester)\((.*)\)')


class OperationInspector:
    def __init__(self, obj):
        """
        Information about operations that can act on a given LibGAP element

        INPUT:

        - ``obj`` -- a `~gappy.element.GapObj`.

        EXAMPLES::

            >>> from gappy.operations import OperationInspector
            >>> OperationInspector(gap(123))
            Operations on 123
        """

        self._obj = obj
        self._gap = obj.parent()

        # TODO: These functions/globals were originally module-level globals
        # instantiated from the global libgap instance.
        # Currently that is fine, since there can only ever be one GAP
        # interpreter instance at the moment, but with an eye toward supporting
        # multiple GAP interpreters and general refactoring, these are moved to
        # instance-level variables.  However, we could still speed this up by
        # caching these somewhere on a per-Gap basis (function_factory is
        # already supposed to be cached so that alone might be good enough once
        # caching is restored on it).
        FlagsType = self._gap.function_factory('FlagsType')
        TypeObj = self._gap.function_factory('TypeObj')

        self.flags = FlagsType(TypeObj(self.obj))

    def __repr__(self):
        """
        Return the string representation

        OUTPUT:

        String

        EXAMPLES::

            >>> from gappy.operations import OperationInspector
            >>> opr = OperationInspector(gap(123))
            >>> opr.__repr__()
            'Operations on 123'
        """
        return 'Operations on {0}'.format(repr(self._obj))

    @property
    def obj(self):
        """
        The first argument for the operations

        OUTPUT:

        A Libgap object.

        EXAMPLES::

            >>> from gappy.operations import OperationInspector
            >>> x = OperationInspector(gap(123))
            >>> print(x.obj)
            123
        """
        return self._obj

    def operations(self):
        """
        Return the GAP operations for :meth:`obj`

        OUTPUT:

        List of GAP operations

        EXAMPLES::

            >>> from gappy.operations import OperationInspector
            >>> x = OperationInspector(gap(123))
            >>> Unknown = gap.function_factory('Unknown')
            >>> Unknown in x.operations()
            True
        """
        IS_SUBSET_FLAGS = self._gap.function_factory('IS_SUBSET_FLAGS')
        GET_OPER_FLAGS = self._gap.function_factory('GET_OPER_FLAGS')
        OPERATIONS = self._gap.get_global('OPERATIONS')

        def mfi(o):
            filts = GET_OPER_FLAGS(o)
            return any(all(IS_SUBSET_FLAGS(self.flags, fl) for fl in fls)
                       for fls in filts)

        return (op for op in OPERATIONS if mfi(op))

    def op_names(self):
        """
        Return the names of the operations

        OUTPUT:

        List of strings

        EXAMPLES::

            >>> from gappy.operations import OperationInspector
            >>> x = OperationInspector(gap(123))
            >>> 'Sqrt' in x.op_names()
            True
        """
        NameFunction = self._gap.function_factory('NameFunction')
        result = set()
        for f in self.operations():
            name = NameFunction(f).sage()
            if name[0] not in string.ascii_letters:
                continue
            match = NAME_RE.match(name)
            if match:
                result.add(match.groups()[1])
            else:
                result.add(name)
        return sorted(result)
