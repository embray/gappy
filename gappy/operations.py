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


__all__ = ['OperationInspector']


_NAME_RE = re.compile(r'(Setter|Getter|Tester)\((.*)\)')


class OperationInspector:
    def __init__(self, obj):
        """
        Information about operations that can act on a given GAP object.

        Parameters
        ----------

        obj : `~gappy.element.GapObj`
            A `~gappy.element.GapObj` to query.

        Examples
        --------

        >>> from gappy.operations import OperationInspector
        >>> OperationInspector(gap(123))
        Operations on 123
        """

        self._obj = obj
        self._gap = obj.parent()
        self.flags = self._gap.FlagsType(self._gap.TypeObj(self.obj))

    def __repr__(self):
        """
        Return the string representation

        Returns
        -------

        str

        Examples
        --------

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

        Returns
        -------

        `GapObj`

        Examples
        --------

        >>> from gappy.operations import OperationInspector
        >>> x = OperationInspector(gap(123))
        >>> print(x.obj)
        123
        """
        return self._obj

    def operations(self):
        """
        Return the GAP operations for :meth:`obj`

        Returns
        -------

        generator
            Generator iterating over all operations.

        Examples
        --------

        >>> from gappy.operations import OperationInspector
        >>> x = OperationInspector(gap(123))
        >>> gap.Unknown in x.operations()
        True
        """
        IS_SUBSET_FLAGS = self._gap.IS_SUBSET_FLAGS
        GET_OPER_FLAGS = self._gap.GET_OPER_FLAGS
        OPERATIONS = self._gap.OPERATIONS

        def mfi(o):
            filts = GET_OPER_FLAGS(o)
            return any(all(IS_SUBSET_FLAGS(self.flags, fl) for fl in fls)
                       for fls in filts)

        return (op for op in OPERATIONS if mfi(op))

    def op_names(self):
        """
        Return the names of the operations.

        Returns
        -------
        list
            Sorted list of names.

        Examples
        --------

        >>> from gappy.operations import OperationInspector
        >>> x = OperationInspector(gap(123))
        >>> 'Sqrt' in x.op_names()
        True
        """
        NameFunction = self._gap.NameFunction
        result = set()
        for f in self.operations():
            name = str(NameFunction(f))
            if name[0] not in string.ascii_letters:
                continue
            match = _NAME_RE.match(name)
            if match:
                result.add(match.groups()[1])
            else:
                result.add(name)
        return sorted(result)
