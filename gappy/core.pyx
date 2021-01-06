"""
Utility functions for GAP
"""

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
from posix.dlfcn cimport dlopen, dlclose, dlerror, RTLD_NOW, RTLD_GLOBAL
from cpython.exc cimport PyErr_Fetch, PyErr_Restore
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF
from cysignals.signals cimport sig_on, sig_off

import locale
import os
import warnings
import sys

from .gap_includes cimport *
from .element cimport *
from .exceptions import GAPError
from .utils import get_gap_memory_pool_size

#from sage.cpython.string cimport str_to_bytes, char_to_str
cdef str_to_bytes(str s, str encoding='utf-8', str errors='strict'):
    return s.encode(encoding, errors)
cdef char_to_str(char *s):
    return s.decode('utf-8')

#from sage.interfaces.gap_workspace import prepare_workspace_dir

#from sage.structure.parent cimport Parent
cdef class Parent:
    def __init__(self, *args, **kwargs):
        pass

ZZ = object()

#from sage.structure.element cimport Vector
cdef class Vector:
    pass


def cached_method(func):
    return func


def current_randstate():
    pass


cdef extern from "<dlfcn.h>" nogil:
    # Missing from posix.dlfcn since it's a non-standard GNU extension
    int dlinfo(void *, int, void *)


cdef extern from "<link.h>" nogil:
    int RTLD_DI_LINKMAP
    cdef struct link_map:
        char *l_name


# TODO: Linux-specific for now, on MacOS it would by libgap.dylib etc., not
# sure if dlopen on MacOS will make this replacement automatically or not.
# It might be nice if libgap-api.h actually included the expected library
# name as a macro.
LIBGAP_SONAME = "libgap.so"

_FS_ENCODING = sys.getfilesystemencoding()
_LOC_ENCODING = locale.getpreferredencoding()


############################################################################
### Hooking into the GAP memory management #################################
############################################################################


cdef class ObjWrapper(object):
    """
    Wrapper for GAP master pointers

    EXAMPLES::

        >>> from sage.libs.gap.util import ObjWrapper
        >>> x = ObjWrapper()
        >>> y = ObjWrapper()
        >>> x == y
        True
    """

    def __richcmp__(ObjWrapper self, ObjWrapper other, int op):
        r"""
        Comparison wrapped Obj.

        INPUT:

        - ``lhs``, ``rhs`` -- :class:`ObjWrapper`.

        - ``op`` -- integer. The comparison operation to be performed.

        OUTPUT:

        Boolean.

        EXAMPLES::

            >>> from sage.libs.gap.util import ObjWrapper
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

        EXAMPLES::

            >>> from sage.libs.gap.util import ObjWrapper
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
        MarkBag((<ObjWrapper>obj).value)


############################################################################
### Initialization of GAP ##################################################
############################################################################


# To ensure that we call initialize_libgap only once.
cdef bint _gap_is_initialized = False


cdef char* _reset_error_output_cmd = """\
libgap_errout := "";
MakeReadWriteGlobal("ERROR_OUTPUT");
ERROR_OUTPUT := OutputTextString(libgap_errout, false);
MakeReadOnlyGlobal("ERROR_OUTPUT");
"""

cdef char* _close_error_output_cmd = """\
CloseStream(ERROR_OUTPUT);
MakeReadWriteGlobal("ERROR_OUTPUT");
ERROR_OUTPUT := "*errout*";
MakeReadOnlyGlobal("ERROR_OUTPUT");
MakeImmutable(libgap_errout);
"""


# TODO: Change autoload=True by default
cdef initialize(gap_root=None, libgap_soname=None, autoload=False):
    """
    Initialize the GAP library, if it hasn't already been
    initialized.  It is safe to call this multiple times.

    TESTS::

        >>> gap(123)   # indirect doctest
        123
    """
    cdef link_map lm
    cdef int ret
    cdef char *error

    global _gap_is_initialized
    if _gap_is_initialized: return
    # Hack to ensure that all symbols provided by libgap are loaded into the
    # global symbol table
    # Note: we could use RTLD_NOLOAD and avoid the subsequent dlclose() but
    # this isn't portable

    if libgap_soname is None:
        libgap_soname = LIBGAP_SONAME

    cdef void* handle
    handle = dlopen(libgap_soname.encode('ascii'), RTLD_NOW | RTLD_GLOBAL)
    if handle is NULL:
        raise RuntimeError(
                f"Could not dlopen() {libgap_soname} even though it should "
                "already be loaded!")

    if gap_root is None:
        gap_root = os.environ.get('GAP_ROOT')
        if gap_root is None:
            # Use dlinfo to try to determine the path to libgap.so.  If it is
            # from within a GAP_ROOT we can use it; otherwise we will not
            # be able to determine GAP_ROOT
            ret = dlinfo(handle, RTLD_DI_LINKMAP, &lm)
            if ret != 0:
                error = dlerror()
                raise RuntimeError(
                    f'Could not dlinfo() {libgap_soname}: '
                    f'{error.decode(_LOC_ENCODING, "surrogateescape")}; '
                    f'cannot determine path to GAP_ROOT')

            so_path = lm.l_name.decode(_FS_ENCODING, 'surrogateescape')
            gap_root = os.path.dirname(os.path.dirname(so_path))

    dlclose(handle)

    # If gap_root is still None we cannot proceed because GAP actually crashes
    # if we try to do anything without loading GAP's stdlib
    hint = ('Either pass gap_root when initializing the Gap class, '
            'or pass it via the GAP_ROOT environment variable.')
    if gap_root is None:
        raise RuntimeError(f"Could not determine path to GAP_ROOT.  {hint}")
    elif not os.path.exists(os.path.join(gap_root, 'lib', 'init.g')):
        raise RuntimeError(
            f'GAP_ROOT path {gap_root} does not contain lib/init.g which is '
            f'needed for GAP to work.  {hint}')

    # Define argv variable, which we will pass in to
    # initialize GAP. Note that we must pass define the memory pool
    # size!
    cdef char* argv[16]
    cdef int argc = 14

    argv[0] = ''
    argv[1] = '-l'
    s = gap_root.encode(_FS_ENCODING, 'surrogateescape')
    argv[2] = s

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

    # argv[argc] must be NULL
    argv[argc] = NULL

    #from .saved_workspace import workspace
    #workspace, workspace_is_up_to_date = workspace()
    #ws = str_to_bytes(workspace, FS_ENCODING, "surrogateescape")
    #if workspace_is_up_to_date:
    #    argv[argc] = "-L"
    #    argv[argc + 1] = ws
    #    argc += 2

    # Get the path to the sage.gaprc file and check that it exists
    #sage_gaprc = os.path.join(os.path.dirname(__file__), 'sage.gaprc')
    #if not os.path.exists(sage_gaprc):
    #    warnings.warn(f"Sage's GAP initialization file {sage_gaprc} is "
    #                   "is missing; some functionality may be limited")
    #else:
    #    sage_gaprc = str_to_bytes(sage_gaprc, FS_ENCODING, "surrogateescape")
    #    argv[argc] = sage_gaprc
    #    argc += 1

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

    # Save a new workspace if necessary
    #if not workspace_is_up_to_date:
    #    prepare_workspace_dir()
    #    from sage.misc.temporary_file import atomic_write
    #    with atomic_write(workspace) as f:
    #        f.close()
    #        gap_eval('SaveWorkspace("{0}")'.format(f.name))


############################################################################
### Evaluate string in GAP #################################################
############################################################################

cdef Obj gap_eval(str gap_string) except? NULL:
    r"""
    Evaluate a string in GAP.

    INPUT:

    - ``gap_string`` -- string. A valid statement in GAP.

    OUTPUT:

    The resulting GAP object or NULL+Python Exception in case of error.
    The result object may also be NULL without a Python exception set for
    statements that do not return a value.

    EXAMPLES::

        >>> gap.eval('if 4>3 then\nPrint("hi");\nfi')
        >>> gap.eval('1+1')   # testing that we have successfully recovered
        2

        >>> gap.eval('if 4>3 thenPrint("hi");\nfi')
        Traceback (most recent call last):
        ...
        GAPError: Syntax error: then expected in stream:1
        if 4>3 thenPrint("hi");
               ^^^^^^^^^
        >>> gap.eval('1+1')   # testing that we have successfully recovered
        2

    TESTS:

    A bad eval string that results in multiple statement evaluations by GAP
    and hence multiple errors should still result in a single exception
    with a message capturing all errors that occurrer::

        >>> gap.eval('Complex Field with 53 bits of precision;')
        Traceback (most recent call last):
        ...
        GAPError: Error, Variable: 'Complex' must have a value
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
    left in the error stream)::

        >>> gap.eval('Complex Field with 53 bits of precision;')
        Traceback (most recent call last):
        ...
        GAPError: Error, Variable: 'Complex' must have a value
        ...
        Error, Variable: 'precision' must have a value

        >>> gap.eval('1+1')  # test that we successfully recover
        2
    """
    initialize()
    cdef Obj result
    cdef int i, j, nresults

    # Careful: We need to keep a reference to the bytes object here
    # so that Cython doesn't deallocate it before GAP is done with
    # its contents.
    cmd = str_to_bytes(gap_string + ';\n')
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
        nresults = LEN_LIST(result)
        if nresults > 1:  # to mimick the old libGAP
            # TODO: Get rid of this restriction eventually?
            raise GAPError("can only evaluate a single statement")

        # Get the result of the first statement
        result = ELM0_LIST(result, 1) # 1-indexed!

        if ELM0_LIST(result, 1) != GAP_True:
            # An otherwise unhandled error occurred in GAP (such as a
            # syntax error).  Try running the error handler manually
            # to capture the error output, if any.
            # This should result in a GAPError being set.
            error_handler_check_exception()

        # The actual resultant object, if any, is in the second entry
        # (which may be unassigned--see previous github comment; in this case
        # 0 is returned without setting a a Python exception, so we should treat
        # this like returning None)

        return ELM0_LIST(result, 2)
    finally:
        GAP_Leave()
        sig_off()


############################################################################
### Error handler ##########################################################
############################################################################

cdef str extract_libgap_errout():
    """
    Reads the global variable libgap_errout and returns a Python string
    containing the error message (with some boilerplate removed).
    """
    cdef Obj r
    cdef char *msg

    r = GAP_ValueGlobalVariable("libgap_errout")

    # Grab a pointer to the C string underlying the GAP string libgap_errout
    # then copy it to a Python str (char_to_str contains an implicit strcpy)
    msg = CSTR_STRING(r)
    if msg != NULL:
        msg_py = char_to_str(msg)
        msg_py = msg_py.replace('For debugging hints type ?Recovery from '
                                'NoMethodFound\n', '').strip()
    else:
        # Shouldn't happen but just in case...
        msg_py = ""

    return msg_py


cdef void error_handler() with gil:
    """
    The libgap error handler.

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

        # Fetch any existing exception before calling
        # extract_libgap_errout() so that the exception indicator is
        # cleared
        PyErr_Fetch(&exc_type, &exc_val, &exc_tb)

        msg = extract_libgap_errout()
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
# The libGap interpreter object Gap is the parent of the GapElements


class Gap(Parent):
    r"""
    The GAP interpreter object.

    .. NOTE::

        This object must be instantiated exactly once by the
        libgap.  Always use the provided ``libgap`` instance, and never
        instantiate :class:`Gap` manually.

        # TODO: Actually this will change when Gap becomes a singleton class; it
        will be safe to initialize Gap() with alternate arguments from the
        defaults before its first use; after that it cannot be re-initialized.

    EXAMPLES::

        >>> gap.eval('SymmetricGroup(4)')
        Sym( [ 1 .. 4 ] )

    TESTS::

        >>> TestSuite(gap).run(skip=['_test_category', '_test_elements', '_test_pickling'])
    """

    Element = GapElement

    def _coerce_map_from_(self, S):
        """
        Whether a coercion from `S` exists.

        INPUT / OUTPUT:

        See :mod:`sage.structure.parent`.

        EXAMPLES::

            >>> gap.has_coerce_map_from(ZZ)
            True
            >>> gap.has_coerce_map_from(CyclotomicField(5)['x','y'])
            True
        """
        return True

    def _element_constructor_(self, x):
        r"""
        Construct elements of this parent class.

        INPUT:

        - ``x`` -- anything that defines a GAP object.

        OUTPUT:

        A :class:`GapElement`.

        EXAMPLES::

            >>> gap(0)   # indirect doctest
            0
            >>> gap(ZZ(0))
            0
            >>> gap(int(0))
            0
            >>> gap(vector((0,1,2)))
            [ 0, 1, 2 ]
            >>> gap(vector((1/3,2/3,4/5)))
            [ 1/3, 2/3, 4/5 ]
            >>> gap(vector((1/3, 0.8, 3)))
            [ 0.333333, 0.8, 3. ]

        """
        initialize()
        if isinstance(x, GapElement):
            return x
        elif isinstance(x, (list, tuple, Vector)):
            return make_GapElement_List(self, make_gap_list(self, x))
        elif isinstance(x, dict):
            return make_GapElement_Record(self, make_gap_record(self, x))
        elif isinstance(x, bool):
            # attention: must come before int
            return make_GapElement_Boolean(self, GAP_True if x else GAP_False)
        elif isinstance(x, int):
            return make_GapElement_Integer(self, make_gap_integer(x))
        elif isinstance(x, basestring):
            return make_GapElement_String(self, make_gap_string(x))
        else:
            try:
                return x._libgap_()
            except AttributeError:
                pass
            x = str(x._libgap_init_())
            return make_any_gap_element(self, gap_eval(x))

    def _construct_matrix(self, M):
        """
        Construct a LibGAP matrix.

        INPUT:

        - ``M`` -- a matrix.

        OUTPUT:

        A GAP matrix, that is, a list of lists with entries over a
        common ring.

        EXAMPLES::

            >>> M = gap._construct_matrix(identity_matrix(ZZ,2)); M
            [ [ 1, 0 ], [ 0, 1 ] ]
            >>> M.IsMatrix()
            true

            >>> M = gap(identity_matrix(ZZ,2)); M  # syntactic sugar
            [ [ 1, 0 ], [ 0, 1 ] ]
            >>> M.IsMatrix()
            true

            >>> M = gap(matrix(GF(3),2,2,[4,5,6,7])); M
            [ [ Z(3)^0, Z(3) ], [ 0*Z(3), Z(3)^0 ] ]
            >>> M.IsMatrix()
            true

            >>> x = polygen(QQ, 'x')
            >>> M = gap(matrix(QQ['x'],2,2,[x,5,6,7])); M
            [ [ x, 5 ], [ 6, 7 ] ]
            >>> M.IsMatrix()
            true

        TESTS:

        We gracefully handle the case that the conversion fails (:trac:`18039`)::

            >>> F.<a> = GF(9, modulus="first_lexicographic")
            >>> gap(Matrix(F, [[a]]))
            Traceback (most recent call last):
            ...
            NotImplementedError: conversion of (Givaro) finite field element to GAP not implemented except for fields defined by Conway polynomials.
        """
        ring = M.base_ring()
        try:
            gap_ring = self(ring)
        except ValueError:
            raise TypeError('base ring is not supported by GAP')
        M_list = map(list, M.rows())
        return make_GapElement_List(self, make_gap_matrix(self, M_list, gap_ring))

    def eval(self, gap_command):
        """
        Evaluate a gap command and wrap the result.

        INPUT:

        - ``gap_command`` -- a string containing a valid gap command
          without the trailing semicolon.

        OUTPUT:

        A :class:`GapElement`.

        EXAMPLES::

            >>> gap.eval('0')
            0
            >>> gap.eval('"string"')
            "string"
        """
        cdef GapElement elem

        if not isinstance(gap_command, basestring):
            gap_command = str(gap_command._libgap_init_())

        initialize()
        elem = make_any_gap_element(self, gap_eval(gap_command))

        # If the element is NULL just return None instead
        if elem.value == NULL:
            return None

        return elem

    def load_package(self, pkg):
        """
        If loading fails, raise a RuntimeError exception.

        TESTS::

            >>> gap.load_package("chevie")
            Traceback (most recent call last):
            ...
            RuntimeError: Error loading GAP package chevie. You may want to
            install gap_packages SPKG.
        """
        load_package = self.function_factory('LoadPackage')
        # Note: For some reason the default package loading error messages are
        # controlled with InfoWarning and not InfoPackageLoading
        prev_infolevel = self.InfoLevel(self.InfoWarning)
        self.SetInfoLevel(self.InfoWarning, 0)
        ret = load_package(pkg)
        self.SetInfoLevel(self.InfoWarning, prev_infolevel)
        if str(ret) == 'fail':
            raise RuntimeError(f"Error loading GAP package {pkg}.  "
                               f"You may want to install gap_packages SPKG.")
        return ret

    @cached_method
    def function_factory(self, function_name):
        """
        Return a GAP function wrapper

        This is almost the same as calling
        ``gap.eval(function_name)``, but faster and makes it
        obvious in your code that you are wrapping a function.

        INPUT:

        - ``function_name`` -- string. The name of a GAP function.

        OUTPUT:

        A function wrapper
        :class:`~sage.libs.gap.element.GapElement_Function` for the
        GAP function. Calling it from Sage is equivalent to calling
        the wrapped function from GAP.

        EXAMPLES::

            >>> gap.function_factory('Print')
            <Gap function "Print">
        """
        initialize()
        return make_GapElement_Function(self, gap_eval(function_name))

    def set_global(self, variable, value):
        """
        Set a GAP global variable

        INPUT:

        - ``variable`` -- string. The variable name.

        - ``value`` -- anything that defines a GAP object.

        EXAMPLES::

            >>> gap.set_global('FooBar', 1)
            >>> gap.get_global('FooBar')
            1
            >>> gap.unset_global('FooBar')
            >>> gap.get_global('FooBar')
            Traceback (most recent call last):
            ...
            GAPError: Error, VAL_GVAR: No value bound to FooBar
        """
        is_bound = self.function_factory('IsBoundGlobal')
        bind_global = self.function_factory('BindGlobal')
        if is_bound(variable):
            self.unset_global(variable)
        bind_global(variable, value)

    def unset_global(self, variable):
        """
        Remove a GAP global variable

        INPUT:

        - ``variable`` -- string. The variable name.

        EXAMPLES::

            >>> gap.set_global('FooBar', 1)
            >>> gap.get_global('FooBar')
            1
            >>> gap.unset_global('FooBar')
            >>> gap.get_global('FooBar')
            Traceback (most recent call last):
            ...
            GAPError: Error, VAL_GVAR: No value bound to FooBar
        """
        is_readonlyglobal = self.function_factory('IsReadOnlyGlobal')
        make_readwrite = self.function_factory('MakeReadWriteGlobal')
        unbind_global = self.function_factory('UnbindGlobal')
        if is_readonlyglobal(variable):
            make_readwrite(variable)
        unbind_global(variable)

    def get_global(self, variable):
        """
        Get a GAP global variable

        INPUT:

        - ``variable`` -- string. The variable name.

        OUTPUT:

        A :class:`~sage.libs.gap.element.GapElement` wrapping the GAP
        output. A ``ValueError`` is raised if there is no such
        variable in GAP.

        EXAMPLES::

            >>> gap.set_global('FooBar', 1)
            >>> gap.get_global('FooBar')
            1
            >>> gap.unset_global('FooBar')
            >>> gap.get_global('FooBar')
            Traceback (most recent call last):
            ...
            GAPError: Error, VAL_GVAR: No value bound to FooBar
        """
        value_global = self.function_factory('ValueGlobal')
        return value_global(variable)

    def global_context(self, variable, value):
        """
        Temporarily change a global variable

        INPUT:

        - ``variable`` -- string. The variable name.

        - ``value`` -- anything that defines a GAP object.

        OUTPUT:

        A context manager that sets/reverts the given global variable.

        EXAMPLES::

            >>> gap.set_global('FooBar', 1)
            >>> with gap.global_context('FooBar', 2):
            ...     print(gap.get_global('FooBar'))
            2
            >>> gap.get_global('FooBar')
            1
        """
        from sage.libs.gap.context_managers import GlobalVariableContext
        initialize()
        return GlobalVariableContext(variable, value)

    def set_seed(self, seed=None):
        """
        Reseed the standard GAP pseudo-random sources with the given seed.

        Uses a random seed given by ``current_randstate().ZZ_seed()`` if
        ``seed=None``.  Otherwise the seed should be an integer.

        EXAMPLES::

            >>> gap.set_seed(0)
            0
            >>> [gap.Random(1, 10) for i in range(5)]
            [2, 3, 3, 4, 2]
        """
        if seed is None:
            seed = current_randstate().ZZ_seed()

        Reset = self.function_factory("Reset")
        Reset(self.GlobalMersenneTwister, seed)
        Reset(self.GlobalRandomSource, seed)
        return seed

    def _an_element_(self):
        r"""
        Return a :class:`GapElement`.

        OUTPUT:

        A :class:`GapElement`.

        EXAMPLES::

            >>> gap.an_element()   # indirect doctest
            0
        """
        return self(0)

    def zero(self):
        """
        Return (integer) zero in GAP.

        OUTPUT:

        A :class:`GapElement`.

        EXAMPLES::

            >>> gap.zero()
            0
        """
        return self(0)

    def one(self):
        r"""
        Return (integer) one in GAP.

        EXAMPLES::

            >>> gap.one()
            1
            >>> parent(_)
            C library interface to GAP
        """
        return self(1)

    def __init__(self):
        r"""
        The Python constructor.

        EXAMPLES::

            >>> type(gap)
            <type 'sage.misc.lazy_import.LazyImport'>
            >>> type(gap._get_object())
            <class 'sage.libs.gap.gap.Gap'>
        """
        Parent.__init__(self, base=ZZ)

    def __repr__(self):
        r"""
        Return a string representation of ``self``.

        OUTPUT:

        String.

        EXAMPLES::

            >>> gap
            C library interface to GAP
        """
        return 'C library interface to GAP'

    @cached_method
    def __dir__(self):
        """
        Customize tab completion

        EXAMPLES::

           >>> 'OctaveAlgebra' in dir(gap)
           True
        """
        from sage.libs.gap.gap_globals import common_gap_globals
        return dir(self.__class__) + sorted(common_gap_globals)

    def __getattr__(self, name):
        r"""
        The attributes of the Gap object are the Gap functions, and in some
        cases other global variables from GAP.

        INPUT:

        - ``name`` -- string. The name of the GAP function you want to
          call.

        OUTPUT:

        A :class:`GapElement`. A ``AttributeError`` is raised
        if there is no such function or global variable.

        EXAMPLES::

            >>> gap.List
            <Gap function "List">
            >>> gap.GlobalRandomSource
            <RandomSource in IsGlobalRandomSource>
        """
        if name in dir(self.__class__):
            return getattr(self.__class__, name)

        try:
            g = self.eval(name)
        except ValueError:
            raise AttributeError(f'No such attribute: {name}.')

        self.__dict__[name] = g
        return g

    def show(self):
        """
        Return statistics about the GAP owned object list

        This includes the total memory allocated by GAP as returned by
        ``gap.eval('TotalMemoryAllocated()'), as well as garbage collection
        / object count statistics as returned by
        ``gap.eval('GasmanStatistics')``, and finally the total number of
        GAP objects held by Sage as :class:`~sage.libs.gap.element.GapElement`
        instances.

        The value ``livekb + deadkb`` will roughly equal the total memory
        allocated for GAP objects (see
        ``gap.eval('TotalMemoryAllocated()')``).

        .. note::

            Slight complication is that we want to do it without accessing
            libgap objects, so we don't create new GapElements as a side
            effect.

        EXAMPLES::

            >>> a = gap(123)
            >>> b = gap(456)
            >>> c = gap(789)
            >>> del b
            >>> gap.collect()
            >>> gap.show()  # random output
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
        d['total_alloc'] = self.eval('TotalMemoryAllocated()').sage()
        d['gasman_stats'] = self.eval('GasmanStatistics()').sage()
        return d

    def count_GAP_objects(self):
        """
        Return the number of GAP objects that are being tracked by
        GAP.

        OUTPUT:

        An integer

        EXAMPLES::

            >>> gap.count_GAP_objects()   # random output
            5
        """
        return len(get_owned_objects())

    def collect(self):
        """
        Manually run the garbage collector

        EXAMPLES::

            >>> a = gap(123)
            >>> del a
            >>> gap.collect()
        """
        initialize()
        rc = CollectBags(0, 1)
        if rc != 1:
            raise RuntimeError('Garbage collection failed.')


gap = Gap()
