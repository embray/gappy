"""Top-level Python interface to GAP."""

#*****************************************************************************
#       Copyright (C) 2012 Volker Braun <vbraun.name@gmail.com>
#       Copyright (C) 2021 E. Madison Bray <embray@lri.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from libc.signal cimport signal, SIGCHLD, SIG_DFL
from cpython.exc cimport PyErr_Fetch, PyErr_Restore
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cysignals.signals cimport sig_on, sig_off

import locale
import os
import re
import sys
import warnings
from functools import lru_cache
from numbers import Integral, Rational, Real

from .context_managers import GlobalVariableContext
from .exceptions import GAPError
from .gap_globals import common_gap_globals as GAP_GLOBALS
from .gap_includes cimport *
from .gapobj cimport *
from .gmp cimport *
from .utils import (get_gap_memory_pool_size, get_gap_root, _SPECIAL_ATTRS,
        _FS_ENCODING)


############################################################################
### Hooking into the GAP memory management #################################
############################################################################


cdef class ObjWrapper(object):
    """
    Wrapper for GAP master pointers

    Examples
    --------

    >>> from gappy.core import ObjWrapper
    >>> x = ObjWrapper()
    >>> y = ObjWrapper()
    >>> x == y
    True
    """

    def __richcmp__(ObjWrapper self, ObjWrapper other, int op):
        r"""
        Comparison wrapped Obj.

        Parameters
        ----------

        other : `ObjWrapper`
            The other `ObjWrapper` to compare to.

        op : int
            The comparison operation to be performed.

        Returns
        -------

        bool
            The result of the comparison.

        Examples
        --------

        >>> from gappy.core import ObjWrapper
        >>> x = ObjWrapper()
        >>> y = ObjWrapper()
        >>> x == y
        True
        """
        cdef result
        cdef Obj self_value = self.value
        cdef Obj other_value = other.value
        if op == Py_LT:
            return self_value < other_value
        elif op == Py_LE:
            return self_value <= other_value
        elif op == Py_EQ:
            return self_value == other_value
        elif op == Py_GT:
            return self_value > other_value
        elif op == Py_GE:
            return self_value >= other_value
        elif op == Py_NE:
            return self_value != other_value
        else:
            assert False  # unreachable

    def __hash__(self):
        """
        Return a hash value

        Examples
        --------

        >>> from gappy.core import ObjWrapper
        >>> x = ObjWrapper()
        >>> hash(x)
        0
        """
        return <Py_hash_t>(self.value)


cdef ObjWrapper wrap_obj(Obj obj):
    """
    Constructor function for :class:`ObjWrapper`
    """
    cdef ObjWrapper result = ObjWrapper.__new__(ObjWrapper)
    result.value = obj
    return result


# a dictionary to keep all GAP elements
# needed for GASMAN callbacks
#
cdef dict owned_objects_refcount = dict()

#
# used in Gap.count_GAP_objects
#
cpdef get_owned_objects():
    """
    Helper to access the refcount dictionary from Python code
    """
    return owned_objects_refcount


cdef void reference_obj(Obj obj):
    """
    Reference ``obj``
    """
    cdef ObjWrapper wrapped = wrap_obj(obj)
    global owned_objects_refcount
#    print("reference_obj called "+ crepr(obj) +"\n")
    if wrapped in owned_objects_refcount:
        owned_objects_refcount[wrapped] += 1
    else:
        owned_objects_refcount[wrapped] = 1


cdef void dereference_obj(Obj obj):
    """
    Reference ``obj``
    """
    cdef ObjWrapper wrapped = wrap_obj(obj)
    global owned_objects_refcount
    refcount = owned_objects_refcount.pop(wrapped)
    if refcount > 1:
        owned_objects_refcount[wrapped] = refcount - 1


cdef void gasman_callback() with gil:
    """
    Callback before each GAP garbage collection
    """
    global owned_objects_refcount
    for obj in owned_objects_refcount:
        GAP_MarkBag((<ObjWrapper>obj).value)


############################################################################
### Initialization of GAP ##################################################
############################################################################


# To ensure that we call initialize_libgap only once.
cdef bint _gap_is_initialized = False
cdef Gap _gap_instance = None


cdef char* _reset_error_output_cmd = r"""\
\$GAPPY_ERROUT := "";
MakeReadWriteGlobal("ERROR_OUTPUT");
ERROR_OUTPUT := OutputTextString(\$GAPPY_ERROUT, false);
MakeReadOnlyGlobal("ERROR_OUTPUT");
"""

cdef char* _close_error_output_cmd = """\
CloseStream(ERROR_OUTPUT);
MakeReadWriteGlobal("ERROR_OUTPUT");
ERROR_OUTPUT := "*errout*";
MakeReadOnlyGlobal("ERROR_OUTPUT");
MakeImmutable(\$GAPPY_ERROUT);
"""


# TODO: Change autoload=True by default
cdef initialize(gap_root=None, gaprc=None, workspace=None, autoload=False):
    """
    Initialize the GAP library, if it hasn't already been initialized.

    It is safe to call this multiple times.
    """

    global _gap_is_initialized

    if _gap_is_initialized:
        return

    gap_root = get_gap_root(gap_root=gap_root)

    # Define argv variable, which we will pass in to
    # initialize GAP. Note that we must pass define the memory pool
    # size!
    cdef char* argv[19]
    cdef int argc = 14

    argv[0] = ''
    argv[1] = '-l'
    _gap_root = gap_root.encode(_FS_ENCODING, 'surrogateescape')
    argv[2] = _gap_root

    memory_pool = get_gap_memory_pool_size().encode('ascii')
    argv[3] = '-o'
    argv[4] = memory_pool
    argv[5] = '-s'
    argv[6] = memory_pool

    argv[7] = '-m'
    argv[8] = '64m'

    argv[9] = '-q'    # no prompt!
    argv[10] = '-E'   # don't use readline as this will interfere with Python
    argv[11] = '--nointeract'  # Implies -T
    argv[12] = '-x'    # set the "screen" width so that GAP is less likely to
    argv[13] = '4096'  # insert newlines when printing objects
                       # 4096 unfortunately is the hard-coded max, but should
                       # be long enough for most cases

    if not autoload:
        argv[argc] = '-A'
        argc += 1

    if workspace is not None:
        # Try opening the workspace file, raising the appropriate OSError
        # if not found/readable
        workspace = os.path.normpath(workspace)

        with open(workspace, 'rb'):
            pass

        workspace_ = workspace.encode(_FS_ENCODING, 'surrogateescape')
        argv[argc] = "-L"
        argv[argc + 1] = workspace_
        argc += 2

    if gaprc is not None:
        # Try opening the gaprc file, raising the appropriate OSError
        # if not found/readable
        gaprc = os.path.normpath(gaprc)

        with open(gaprc, 'rb'):
            pass

        gaprc_ = gaprc.encode(_FS_ENCODING, 'surrogateescape')
        argv[argc] = gaprc_
        argc += 1

    # argv[argc] must be NULL
    argv[argc] = NULL

    sig_on()
    # Initialize GAP but disable their SIGINT handler
    GAP_Initialize(argc, argv, gasman_callback, error_handler,
                   handleSignals=False)
    sig_off()

    # Disable GAP's SIGCHLD handler ChildStatusChanged(), which calls
    # waitpid() on random child processes.
    signal(SIGCHLD, SIG_DFL)

    # Set the ERROR_OUTPUT global in GAP to an output stream in which to
    # receive error output
    GAP_EvalString(_reset_error_output_cmd)

    # Finished!
    _gap_is_initialized = True

    # Return a dict of the final initialization args (after handling defaults)
    return {
        'gap_root': gap_root,
        'gaprc': gaprc,
        'workspace': workspace,
        'autoload': autoload
    }


############################################################################
### Evaluate string in GAP #################################################
############################################################################

cdef Obj gap_eval(str gap_string) except? NULL:
    r"""
    Evaluate a string in GAP.

    Parameters
    ----------

    gap_string : str
        A valid statement in GAP.

    Returns
    -------
    GapObj
        The resulting GAP object or NULL+Python Exception in case of error.
        The result object may also be NULL without a Python exception set for
        statements that do not return a value.

    Raises
    ------
    GAPError
        If there was any error in evaluating the statement, be it a syntax
        error, an error in the arguments, etc.

    Examples
    --------

    >>> gap.eval('if 4>3 then\nPrint("hi");\nfi')
    >>> gap.eval('1+1')   # testing that we have successfully recovered
    2

    >>> gap.eval('if 4>3 thenPrint("hi");\nfi')
    Traceback (most recent call last):
    ...
    gappy.exceptions.GAPError: Syntax error: then expected in stream:1
    if 4>3 thenPrint("hi");
           ^^^^^^^^^
    >>> gap.eval('1+1')   # testing that we have successfully recovered
    2

    Tests
    ^^^^^

    A bad eval string that results in multiple statement evaluations by GAP
    and hence multiple errors should still result in a single exception
    with a message capturing all errors that occurrer:

    >>> gap.eval('Complex Field with 53 bits of precision;')
    Traceback (most recent call last):
    ...
    gappy.exceptions.GAPError: Error, Variable: 'Complex' must have a value
    Syntax error: ; expected in stream:1
    Complex Field with 53 bits of precision;;
     ^^^^^^^^^^^^
    Error, Variable: 'with' must have a value
    Syntax error: ; expected in stream:1
    Complex Field with 53 bits of precision;;
     ^^^^^^^^^^^^^^^^^^^^
    Error, Variable: 'bits' must have a value
    Syntax error: ; expected in stream:1
    Complex Field with 53 bits of precision;;
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Error, Variable: 'precision' must have a value

    Test that on a subsequent attempt we get the same message (no garbage was
    left in the error stream):

    >>> gap.eval('Complex Field with 53 bits of precision;')
    Traceback (most recent call last):
    ...
    gappy.exceptions.GAPError: Error, Variable: 'Complex' must have a value
    ...
    Error, Variable: 'precision' must have a value

    >>> gap.eval('1+1')  # test that we successfully recover
    2
    """

    cdef Obj result
    cdef int i, j, nresults
    cdef bytes cmd

    # Careful: We need to keep a reference to the bytes object here
    # so that Cython doesn't deallocate it before GAP is done with
    # its contents.
    cmd = (gap_string.strip() + ';\n').encode('utf-8')
    sig_on()
    try:
        GAP_Enter()
        result = GAP_EvalString(cmd)
        # We can assume that the result object is a GAP PList (plain list)
        # and we should use functions for PLists directly for now; see
        # https://github.com/gap-system/gap/pull/2988/files#r233021437

        # If an error occurred in GAP_EvalString we won't even get
        # here if the error handler was set; but in case it wasn't
        # let's still check the result...
        nresults = GAP_LenList(result)
        if nresults > 1:  # to mimick the old libGAP
            # TODO: Get rid of this restriction eventually?
            raise GAPError("can only evaluate a single statement")

        # Get the result of the first statement
        result = GAP_ElmList(result, 1) # 1-indexed!

        if GAP_ElmList(result, 1) != GAP_True:
            # An otherwise unhandled error occurred in GAP (such as a
            # syntax error).  Try running the error handler manually
            # to capture the error output, if any.
            # This should result in a GAPError being set.
            error_handler_check_exception()

        # The actual resultant object, if any, is in the second entry
        # (which may be unassigned--see previous github comment; in this case
        # 0 is returned without setting a a Python exception, so we should treat
        # this like returning None)

        return GAP_ElmList(result, 2)
    finally:
        GAP_Leave()
        sig_off()


############################################################################
### Error handler ##########################################################
############################################################################

cdef str extract_errout():
    """
    Reads the global variable $GAPPY_ERROUT and returns a Python string
    containing the error message (with some boilerplate removed).
    """
    cdef Obj r
    cdef char *msg

    r = GAP_ValueGlobalVariable("$GAPPY_ERROUT")

    # Grab a pointer to the C string underlying the GAP string $GAPPY_ERROUT
    # then copy it to a Python str
    msg = GAP_CSTR_STRING(r)
    if msg != NULL:
        msg_py = msg.decode('utf-8', 'surrogateescape')
        msg_py = msg_py.replace('For debugging hints type ?Recovery from '
                                'NoMethodFound\n', '').strip()
    else:
        # Shouldn't happen but just in case...
        msg_py = ""

    return msg_py


cdef void error_handler() with gil:
    """
    The gappy error handler.

    If an error occurred, we raise a ``GAPError``; when the original
    ``GAP_EvalString`` returns, this exception will be seen.

    TODO: We should probably prevent re-entering this function if we
    are already handling an error; if there is an error in our stream
    handling code below it could result in a stack overflow.
    """
    cdef PyObject* exc_type = NULL
    cdef PyObject* exc_val = NULL
    cdef PyObject* exc_tb = NULL

    try:
        GAP_EnterStack()

        # Close the error stream: this flushes any remaining output and
        # closes the stream for further writing; reset ERROR_OUTPUT to
        # something sane just in case (trying to print to a closed
        # stream segfaults GAP)
        GAP_EvalStringNoExcept(_close_error_output_cmd)

        # Fetch any existing exception before calling extract_errout() so that
        # the exception indicator is cleared
        PyErr_Fetch(&exc_type, &exc_val, &exc_tb)

        msg = extract_errout()
        # Sometimes error_handler() can be called multiple times
        # from a single GAP_EvalString call before it returns.
        # In this case, we just update the exception by appending
        # to the existing exception message
        if exc_type is <PyObject*>GAPError and exc_val is not NULL:
            msg = str(<object>exc_val) + '\n' + msg
        elif not msg:
            msg = "an unknown error occurred in GAP"

        # Raise an exception using PyErr_Restore().
        # This way, we can keep any existing traceback object.
        # Note that we manually need to deal with refcounts here.
        Py_XDECREF(exc_type)
        Py_XDECREF(exc_val)
        exc_type = <PyObject*>GAPError
        exc_val = <PyObject*>msg
        Py_XINCREF(exc_type)
        Py_XINCREF(exc_val)
        PyErr_Restore(exc_type, exc_val, exc_tb)
    finally:
        # Reset ERROR_OUTPUT with a new text string stream
        GAP_EvalStringNoExcept(_reset_error_output_cmd)
        GAP_LeaveStack()


cdef void error_handler_check_exception() except *:
    error_handler()


############################################################################
### Gap  ###################################################################
############################################################################

cdef class Gap:
    r"""
    The GAP interpreter object.

    .. note::

        When initializing this class, it does not immediately initialize the
        underlying GAP interpreter unless passed ``autoinit=True``.  Otherwise
        the GAP interpreter is not initialized until the first `Gap.eval` call
        or the first GAP global lookup.

        The default interpreter instance `gap` is initialized with some default
        parameters, but before its first use you may initialize your own `Gap`
        instance with different parameters.

    Parameters
    ----------

    gap_root : `str` or `pathlib.Path`
        Path to the GAP installation (GAP_ROOT) you want to use for the GAP
        interpreter.  This should be the path containing the ``lib/`` and
        ``pkg/`` directories for the standard GAP libraries.  Equivalent to
        the ``-l`` command-line argument to ``gap``.
    gaprc : `str` or `pathlib.Path`
        A GAP "runtime config" file containing GAP commands to run immediately
        upon GAP interpreter startup.  Equivalent to passing a GAP file to
        ``gap`` on the command-line.
    workspace : `str` or `pathlib.Path`
        An existing GAP workspace to restore upon interpreter startup.
        Equivalent to the ``-L`` command-line argument to ``gap``.
    autoinit : bool
        Immediately initialize the GAP interpreter when initializing this
        `Gap` instance.  Otherwise the interpreter is initialized "lazily"
        when the first interaction with the interpreter is needed (either
        an `~Gap.eval` call or global variable lookup) (default: `False`).
    autoload : bool
        Automatically load the default recommended GAP packages when starting
        the GAP interpreter.  If `False` this is equivalent to passing the
        ``-A`` command-line argument to ``gap`` (default: `False`).

    Attributes
    ----------
    supported_builtins : tuple
        The basic Python types for which gappy has built-in support for
        conversion to equivalent GAP objects; currently: `str`, `bytes`,
        `bool`, `int`, `numbers.Integral`, `numbers.Rational`, `float`,
        `numbers.Real`, `list`, `tuple`, `dict`, `None`.

    Examples
    --------

    >>> gap.eval('SymmetricGroup(4)')
    Sym( [ 1 .. 4 ] )
    """

    def __cinit__(self):
        # NOTE: When updating this please also update the docstring above;
        # unfortunately there is no straightforward way for Sphinx to extract
        # its value automatically.
        self.supported_builtins = (
            str, bytes, bool, int, Integral, Rational, float, Real, list,
            tuple, dict, type(None)
        )
        self._init_kwargs = {}
        self._converter_registry = {}
        gmp_randinit_default(self._gmp_state)

    cpdef initialize(self):
        """
        Manually initialize the underlying GAP interpreter if it is has not
        already been automatically initialized.

        Returns `True` if this initialized the GAP interpreter for the first
        time, or `False` if the interpreter was already initialized.
        """

        global _gap_instance

        if _gap_is_initialized:
            if _gap_instance is not self:
                raise RuntimeError(
                    'a different Gap instance has already been initialized; '
                    'only one Gap instance can be used at a time')
            return False

        self._init_kwargs.update(initialize(
            gap_root=self._init_kwargs['gap_root'],
            gaprc=self._init_kwargs['gaprc'],
            workspace=self._init_kwargs['workspace'],
            autoload=self._init_kwargs['autoload']
        ))
        _gap_instance = self
        return True

    def __init__(self, gap_root=None, gaprc=None, workspace=None,
                 autoinit=False, autoload=False):
        if _gap_is_initialized:
            raise RuntimeError(
                "the GAP interpreter has already been initialized; only one "
                "GAP interpreter may be initialized in the current process")

        self._init_kwargs.update({
            'gap_root': gap_root,
            'gaprc': gaprc,
            'workspace': workspace,
            'autoload': autoload
        })

        if autoinit:
            self.initialize()

    def __call__(self, x):
        r"""
        Construct GapObj instances from a given object that either has a
        registered converter or a ``_gap_`` or ``_gap_init_`` method.

        Parameters
        ----------

        x
            Any Python object that can be converted to a GAP object.  For the
            list of types that have built-in support for conversion to GAP
            objects, see `.supported_builtins`.

        Returns
        -------

        `GapObj`

        Examples
        --------

        >>> gap(0)
        0
        >>> gap([])
        [ ]
        >>> gap({})
        rec( )
        >>> gap(False)
        false
        >>> gap('')
        ""
        >>> gap(None)
        NULL

        A class with a ``_gap_`` method to convert itself to an equivalent
        `~gappy.gapobj.GapObj`; it is also passed the active `Gap` instance:

        >>> class MyGroup:
        ...     def _gap_(self, gap):
        ...         return gap.SymmetricGroup(3)
        ...
        >>> gap(MyGroup())
        Sym( [ 1 .. 3 ] )

        A class with a ``_gap_init_`` method; same concept but returns a string
        containing any arbitrary GAP code for initializing the object:

        >>> class MyGroup2:
        ...     def _gap_init_(self):
        ...         return 'SymmetricGroup(3)'
        ...
        >>> gap(MyGroup2())
        Sym( [ 1 .. 3 ] )

        .. note::

            Both the ``_gap_`` method and any converter function registered
            with `Gap.register_converter` may return either a
            `~gappy.gapobj.GapObj` *or* one of the built-in types in
            `Gap.supported_builtins` which is then in turn converted to the
            appropriate GAP object.

        """
        self.initialize()
        if isinstance(x, GapObj):
            return x
        elif isinstance(x, (str, bytes)):
            return make_GapString(self, make_gap_string(x))
        elif isinstance(x, bool):
            # attention: must come before int
            return make_GapBoolean(self, GAP_True if x else GAP_False)
        elif isinstance(x, (int, Integral)):
            return make_GapInteger(self, make_gap_integer(int(x)))
        elif isinstance(x, Rational):
            num, denom = x.numerator, x.denominator
            # Special hack for Sage which doesn't implement the Rational
            # ABC correctly; see https://trac.sagemath.org/ticket/28234
            if callable(num):
                num = num()
            if callable(denom):
                denom = denom()
            return self(num) / self(denom)
        elif isinstance(x, (list, tuple)):
            return make_GapList(self, make_gap_list(self, x))
        elif isinstance(x, dict):
            return make_GapRecord(self, make_gap_record(self, x))
        elif isinstance(x, (float, Real)):
            return make_GapFloat(self, make_gap_float(float(x)))
        elif x is None:
            return make_GapObj(self, NULL)
        else:
            converter_name = None
            converter = self._converter_registry.get(type(x))
            if converter is None:
                for type_ in self._converter_registry:
                    if isinstance(x, type_):
                        converter = self._converter_registry[type_]
                        break

            if converter is None:
                if hasattr(x, '_gap_'):
                    ret = x._gap_(self)
                elif hasattr(x, '_gap_init_'):
                    ret = self._from_gap_init(x)
                else:
                    raise ValueError(
                        f'could not convert {x} to a GAP object')
            else:
                ret = converter(x, self)

            if not isinstance(ret, GapObj):
                if isinstance(ret, self.supported_builtins):
                    return self.__call__(ret)

                builtins = ', '.join(
                        t.__name__ for t in self.supported_builtins)
                raise ValueError(
                    f'converter for {type(x).__name__} must return an '
                    f'instance of GapObj or one of {builtins}; '
                    f'got {type(ret).__name__}')
            return ret

    cpdef _from_gap_init(self, x):
        """
        Helper for creating `.GapObj`\s from objects with a ``_gap_init_``
        method.
        """

        return make_any_gap_obj(self, gap_eval(str(x._gap_init_())))

    def register_converter(self, cls, converter):
        """
        Register a converter from a Python type to any type of `.GapObj`.

        This allows providing converters for objects that do not have a
        ``_gap_`` or ``_gap_init_`` method.  While it is preferable to use
        the special methods, this allows adapting classes that do not have
        these methods, without having to subclass or wrap them.

        The type must not be one of the types that already have built-in
        converters; see `Gap.supported_builtins`.

        When converting a Python object to a GAP object, first the built-in
        converters are tried.  Then the registry of converters is checked for
        an *exact* type match.  If no exact match is found, an `isinstance`
        check is performed for each type in the registry.

        The converter is any callable which is passed the the object to convert
        and the `Gap` interpreter instance as its first two arguments, and must
        return a `.GapObj` instance.

        .. note::

            Both the ``_gap_`` method and any converter function registered
            with `Gap.register_converter` may return either a
            `~gappy.gapobj.GapObj` *or* one of the built-in types in
            `Gap.supported_builtins` which is then in turn converted to the
            appropriate GAP object.

        Examples
        --------

        Say we have a class ``PermGroup`` which represents a permutation group
        as a list of permutation generators, themselves representing
        permutations as tuples (which happen to have the same syntax as GAP
        permutations, a fact we can exploit).  This example does not give the
        full implementation of such a class, just the general data structure
        to demonstrate implementing the converter function:

        >>> class PermGroup:
        ...     def __init__(self, *gens):
        ...         self.gens = list(gens)
        ...
        >>> def PermGroup_to_gap(group, gap):
        ...     return gap.eval(f'Group({group.gens})')
        ...
        >>> gap.register_converter(PermGroup, PermGroup_to_gap)
        >>> group = PermGroup((1, 2), (1, 2, 3, 4, 5, 6, 7, 8))
        >>> gap(group)
        Group([ (1,2), (1,2,3,4,5,6,7,8) ])
        """

        if isinstance(cls, self.supported_builtins):
            builtin_names = ', '.join(t.__name__
                                      for t in self.supported_builtins)
            raise ValueError(
                f'type must not be a subclass of one of the types with '
                f'built-in converters: {builtin_names}')

        if not isinstance(cls, type):
            raise ValueError('cls must be a Python class')

        if not callable(converter):
            # TODO: Maybe check the signature as well?
            raise ValueError(
                f'{type.__name__} converter {converter} must be callable')

        if cls in self._converter_registry:
            warnings.warn(
                f'{cls} already has a registered converter '
                f'{self._converter_registry[cls]}; it will be replaced by '
                f'{converter}')

        self._converter_registry[cls] = converter

    @property
    def gap_root(self):
        """
        The path to the GAP installation being used for this interpreter
        instance.

        Examples
        --------

        >>> gap.gap_root  # doctest: +IGNORE_OUTPUT
        '/path/to/gap_installation'
        """

        return self._init_kwargs.get('gap_root')

    @property
    def gaprc(self):
        """
        The path to the GAP runtime configuration file being used for this
        interpreter instance.

        Examples
        --------

        >>> gap.gaprc  # doctest: +IGNORE_OUTPUT
        '/path/to/gaprc'
        """

        return self._init_kwargs.get('gaprc')

    @property
    def workspace(self):
        """
        The path to the GAP workspace loaded by this interpreter instance at
        startup.

        Examples
        --------

        >>> gap.workspace  # doctest: +IGNORE_OUTPUT
        '/path/to/gaprc'
        """

        return self._init_kwargs.get('workspace')

    cpdef eval(self, gap_command):
        """
        Evaluate a gap command and wrap the result.

        Parameters
        ----------

        gap_command : str
            A string containing a valid GAP command with or without the
            trailing semicolon.

        Returns
        -------

        `.GapObj`
            The result of the GAP statement.

        Examples
        --------

        >>> gap.eval('0')
        0
        >>> gap.eval('"string"')
        "string"
        """
        cdef GapObj elem

        if not isinstance(gap_command, str):
            gap_command = str(gap_command._gap_init_())

        self.initialize()
        elem = make_any_gap_obj(self, gap_eval(gap_command))

        # If the element is NULL just return None instead
        if elem.value == NULL:
            return None

        return elem

    _gap_function_re = re.compile(r'^\s*function\s*\(\s*\w*(\s*,\s*\w+)*\s*\)',
                                  re.M)
    """
    Regular expression roughly matching the syntax for a GAP function
    declaration ``function([a, ...])``.  Used for `Gap.gap_function`.
    """

    def gap_function(self, func):
        """
        Create GAP functions from decorated Python functions.

        Examples
        --------

        The code for the GAP function is actually written in the Python
        function's docstring like so:

        >>> @gap.gap_function
        ... def one():
        ...     '''
        ...     Returns the multiplicative identity of the ring of integers.
        ...
        ...     function()
        ...         return 1;
        ...     end;
        ...     '''
        ...
        >>> one
        <GAP function "one">
        >>> one()
        1

        Any text in the docstring before the first line beginning the text
        ``function()`` is used as the function's docstring.  Any following
        text is considered part of the function definition:

        >>> one.help()
        'Returns the multiplicative identity of the ring of integers.'

        Note that using this decorator does *not* cause the GAP interpreter
        to be initialized, so it can be used in module or class-level code.
        The GAP interpreter will only be initialized (if needed) the first time
        the function is called.

        Any Python code in the function's body will be disregarded, so this is
        in effect syntactic sugar for:

        >>> one = gap.eval('function() return 1; end;')

        with the difference being that it can be used to pre-define GAP
        functions without invoking the GAP interpreter directly.

        This decorator may also be used on methods in classes.  In this case
        the ``self``--the instance of the class on which it is defined, is
        always passed as the first argument to the GAP function, *if* it has
        a conversion to a GAP type:

        >>> class MyInt(int):
        ...     @gap.gap_function
        ...     def n_partitions(self):
        ...         '''
        ...         Compute the number of integer partitions.
        ...
        ...         function(n)
        ...             local np;
        ...             if n < 0 then
        ...                 Error("must be a non-negative integer");
        ...             fi;
        ...             np:= function(n, m)
        ...                local i, res;
        ...                if n = 0 then
        ...                   return 1;
        ...                fi;
        ...                res:= 0;
        ...                for i in [1..Minimum(n,m)] do
        ...                   res:= res + np(n-i, i);
        ...                od;
        ...                return res;
        ...             end;
        ...             return np(n,n);
        ...         end;
        ...         '''
        ...
        >>> ten = MyInt(10)
        >>> ten.n_partitions()
        42
        """

        match = self._gap_function_re.search(func.__doc__ or '')
        if match is None:
            raise ValueError(
                f'the docstring for {func} does not contain a GAP function '
                f'definition; it should contain one line beginning with '
                f'function() and the rest should be valid GAP code')
        name = func.__name__
        doc = func.__doc__[:match.start()].strip()
        source = func.__doc__[match.start():].strip()
        return self._gap_function(name, doc, source)

    @lru_cache(maxsize=128)
    def _gap_function(self, name, doc, source):
        """
        Internal implementation of `Gap.gap_function` which caches the created
        functions (up to 128) based on their name, docstring, and source.
        """

        return make_GapLazyFunction(self, name, doc, source)

    def load_package(self, pkg):
        """
        If loading fails, raise a RuntimeError exception.

        Examples
        --------

        >>> gap.load_package("chevie")
        Traceback (most recent call last):
        ...
        RuntimeError: Error loading GAP package chevie.
        """
        # Note: For some reason the default package loading error messages are
        # controlled with InfoWarning and not InfoPackageLoading
        prev_infolevel = self.InfoLevel(self.InfoWarning)
        self.SetInfoLevel(self.InfoWarning, 0)
        ret = self.LoadPackage(pkg)
        self.SetInfoLevel(self.InfoWarning, prev_infolevel)
        if str(ret) == 'fail':
            raise RuntimeError(f'Error loading GAP package {pkg}.')
        return ret

    def set_global(self, variable, value, force=False):
        """
        Set a GAP global variable

        Parameters
        ----------

        variable : str
            The GAP global variable name.
        value
            Any `~gappy.gapobj.GapObj` or Python object that can be converted
            to a GAP object.  Passing `None` is equivalent to `Gap.unset_global`.
        force : bool
            If `True`, sets the value of the global even if it is read-only;
            otherwise an `AttributeError` is raised.

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> gap.get_global('FooBar')
        1
        >>> gap.unset_global('FooBar')
        >>> gap.get_global('FooBar') is None
        True
        >>> gap.set_global('FooBar', 1)
        >>> gap.get_global('FooBar')
        1
        >>> gap.set_global('FooBar', None)
        >>> gap.get_global('FooBar') is None
        True
        """

        cdef bytes name

        self.initialize()
        name = variable.encode('utf-8')
        restore_readonly = False

        if not GAP_CanAssignGlobalVariable(name):
            if force:
                self.MakeReadWriteGlobal(name)
                restore_readonly = True
            else:
                raise AttributeError(
                    f'Cannot set read-only GAP global variable {variable}')

        try:
            obj = self(value)
            GAP_AssignGlobalVariable(name, (<GapObj>obj).value)
        finally:
            if restore_readonly:
                self.MakeReadOnlyGlobal(name)

    def unset_global(self, variable):
        """
        Remove a GAP global variable

        Parameters
        ----------

        variable : str
            The GAP global variable name.

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> gap.get_global('FooBar')
        1
        >>> gap.unset_global('FooBar')
        >>> gap.get_global('FooBar') is None
        True
        """

        cdef bytes name

        self.initialize()
        name = variable.encode('utf-8')

        if not GAP_CanAssignGlobalVariable(name):
            raise AttributeError(
                f'Cannot unset read-only GAP global variable {variable}')

        GAP_AssignGlobalVariable(name, NULL)

    cpdef get_global(self, variable):
        """
        Get a GAP global variable

        Parameters
        ----------

        variable : str
            The GAP global variable name.

        Returns
        -------

        `.GapObj` or `None`
            `.GapObj` wrapping the GAP output.  `None` is returned if there is
             no such variable in GAP.

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> gap.get_global('FooBar')
        1
        >>> gap.unset_global('FooBar')
        >>> gap.get_global('FooBar') is None
        True
        """
        cdef Obj obj
        cdef bytes name

        self.initialize()
        name = variable.encode('utf-8')

        try:
            GAP_Enter()
            obj = GAP_ValueGlobalVariable(name)
            if obj == NULL:
                return None

            return make_any_gap_obj(self, obj)
        finally:
            GAP_Leave()

    def global_context(self, variable, value):
        """
        Temporarily change a global variable

        Parameters
        ----------

        variable : str
            The GAP global variable name.
        value
            Any `~gappy.gapobj.GapObj` or Python object that can be converted
            to a GAP object.  Passing `None` is equivalent to `Gap.unset_global`.

        Returns
        -------

        `.GlobalVariableContext`
            A context manager that sets/reverts the given global variable.

        Examples
        --------

        >>> gap.set_global('FooBar', 1)
        >>> with gap.global_context('FooBar', 2):
        ...     print(gap.get_global('FooBar'))
        2
        >>> gap.get_global('FooBar')
        1
        """
        self.initialize()
        return GlobalVariableContext(self, variable, value)

    def set_seed(self, seed=None):
        """
        Reseed the standard GAP pseudo-random sources with the given seed.

        Uses a random 128-bit integer as the seed given by GMP's
        ``mpz_rrandomm`` if ``seed=None``.  Otherwise the seed should be an
        integer.

        Examples
        --------

        >>> gap.set_seed(0)
        0
        >>> [gap.Random(1, 10) for i in range(5)]
        [2, 3, 3, 4, 2]
        """
        cdef mpz_t z_seed
        cdef Obj gap_seed

        self.initialize()

        if seed is None:
            mpz_init(z_seed)
            mpz_rrandomb(z_seed, self._gmp_state, 128)
            gap_seed = GAP_MakeObjInt(<UInt *>mpz_limbs_read(z_seed),
                                      <Int>mpz_size(z_seed))
            seed = make_GapInteger(self, gap_seed)

        Reset = self.Reset
        Reset(self.GlobalMersenneTwister, seed)
        Reset(self.GlobalRandomSource, seed)
        return seed

    def __repr__(self):
        r"""
        Return a string representation of ``self``.

        Returns
        -------

        str

        Examples
        --------

        >>> gap
        <Gap(gap_root='...')>
        """
        return f'<Gap(gap_root={self.gap_root!r})>'

    def __dir__(self):
        """
        Customize tab completion

        Examples
        --------

        >>> 'OctaveAlgebra' in dir(gap)
        True
        """
        return dir(self.__class__) + sorted(GAP_GLOBALS)

    def __getattr__(self, name):
        r"""
        The attributes of the GAP object are the GAP functions, and in some
        cases other global variables from GAP.

        Parameters
        ----------

        name : str
            The name of the GAP function you want to call or another GAP
            global.

        Returns
        -------

        `GapObj`
            A `GapObj` wrapping the specified global variable in GAP. An
            `AttributeError` is raised if there is no such function or global
            variable.

        Raises
        ------

        AttributeError
            The global variable with the name of the attribute is not bound in
            GAP.

        Examples
        --------

        >>> gap.List
        <GAP function "List">
        >>> gap.GlobalRandomSource
        <RandomSource in IsGlobalRandomSource>
        """
        if name in _SPECIAL_ATTRS:
            # Prevent unintended GAP initialization when displaying in IPython
            raise AttributeError(name)

        val = self.get_global(name)
        if val is None:
            raise AttributeError(f'no GAP global variable bound to {name!r}')
        return val

    def show(self):
        """
        Return statistics about the GAP owned object list

        This includes the total memory allocated by GAP as returned by
        ``gap.eval('TotalMemoryAllocated()'), as well as garbage collection
        / object count statistics as returned by
        ``gap.eval('GasmanStatistics')``, and finally the total number of GAP
        objects held by gappy as :class:`~gappy.gapobj.GapObj` instances.

        The value ``livekb + deadkb`` will roughly equal the total memory
        allocated for GAP objects (see
        ``gap.eval('TotalMemoryAllocated()')``).

        .. note::

            Slight complication is that we want to do it without accessing
            GAP objects, so we don't create new GapObjs as a side effect.

        Examples
        --------

        >>> a = gap(123)
        >>> b = gap(456)
        >>> c = gap(789)
        >>> del b
        >>> gap.collect()
        >>> gap.show()  # doctest: +IGNORE_OUTPUT
        {'gasman_stats': {'full': {'cumulative': 110,
           'deadbags': 321400,
           'deadkb': 12967,
           'freekb': 15492,
           'livebags': 396645,
           'livekb': 37730,
           'time': 110,
           'totalkb': 65536},
          'nfull': 1,
          'npartial': 1},
         'nelements': 23123,
         'total_alloc': 3234234}
        """

        d = {'nelements': self.count_GAP_objects()}
        d['total_alloc'] = int(self.eval('TotalMemoryAllocated()'))
        d['gasman_stats'] = dict(self.eval('GasmanStatistics()'))
        return d

    def count_GAP_objects(self):
        """
        Return the number of GAP objects that are being tracked by
        GAP.

        Returns
        -------

        int

        Examples
        --------

        >>> gap.count_GAP_objects()  # doctest: +IGNORE_OUTPUT
        5
        """
        return len(get_owned_objects())

    def collect(self):
        """
        Manually run the garbage collector

        Examples
        --------

        >>> a = gap(123)
        >>> del a
        >>> gap.collect()
        """
        self.initialize()
        rc = GAP_CollectBags(1)
        if rc != 1:
            raise RuntimeError('Garbage collection failed.')


gap = Gap()
