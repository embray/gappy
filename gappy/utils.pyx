# ****************************************************************************
#       Copyright (C) 2006 William Stein <wstein@gmail.com>
#       Copyright (C) 2021 E. Madison Bray <embray@lri.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import locale
import os
import resource
import sys

from posix.dlfcn cimport dlopen, dlsym, dlclose, dlerror, RTLD_NOW, RTLD_GLOBAL

import psutil


# _repr_mimebundle_ should not be needed here; this seems to be a bug in
# Ipython
_SPECIAL_ATTRS = set([
    '_ipython_canary_method_should_not_exist_',
    '_repr_mimebundle_'
])
"""
Special attributes which should not be looked up in GAP.

Mostly intended to prevent IPython's custom display hook from unintentially
initializing the GAP interpreter.
"""


_FS_ENCODING = sys.getfilesystemencoding()
_LOC_ENCODING = locale.getpreferredencoding()


cdef extern from "<dlfcn.h>" nogil:
    ctypedef struct Dl_info:
        const char *dli_fname

    # dladdr() is non-POSIX, but it appears to be available on Linux, MacOS,
    # BSD and Cygwin at a minimum
    int dladdr(const void *, Dl_info *)


cpdef get_gap_root(gap_root=None):
    """
    Return the path to the GAP installation directory, or "GAP root" where
    the GAP standard library and standard packages are installed.

    Raises an `RuntimeError` if the "GAP root" cannot be found or an `OSError`
    if a system error occurs in the process of searching for it.

    Parameters
    ----------

    gap_root : str or `pathlib.Path`
        Optional user-defined path to the GAP root; if given this foregoes
        other searches and just checks that it looks like a valid GAP root
        (the ``lib/init.g`` file can be found, specifically).

    Examples
    --------

    The exact output of this example will depend on where libgap is installed
    on your system:

    >>> from gappy.utils import get_gap_root
    >>> get_gap_root()  # doctest: +IGNORE_OUTPUT
    '/usr/local/lib/libgap.so'
    """

    cdef void *handle
    cdef void *addr
    cdef char *error
    cdef Dl_info info

    # This code could easily be made more generic, but since it's really only
    # needed for libgap we hard-code the expected library filenames, as well
    # as the known external symbol to look up.
    if sys.platform.startswith('linux') or 'bsd' in sys.platform:
        dylib_name = b'libgap.so'
    elif sys.platform == 'darwin':
        dylib_name = b'libgap.dylib'
    elif sys.platform == 'cygwin':
        dylib_name = b'cyggap-0.dll'
    else:
        # platform not supported
        raise RuntimeError(f'platform not supported by gappy: {sys.platform}')

    # Hack to ensure that all symbols provided by libgap are loaded into the
    # global symbol table
    # Note: we could use RTLD_NOLOAD and avoid the subsequent dlclose() but
    # this isn't portable
    handle = dlopen(dylib_name, RTLD_NOW | RTLD_GLOBAL)
    if handle is NULL:
        error = dlerror()
        raise OSError(
            f'could not open the libgap dynamic library {dylib_name}: '
            f'{error.decode(_LOC_ENCODING, "surrogateescape")}')

    if gap_root is None:
        gap_root = os.environ.get('GAP_ROOT')
        if gap_root is None:
            # Use dlsym() to get the address of a known exported symbol in
            # libgap
            addr = dlsym(handle, b'GAP_Initialize')
            if addr is NULL:
                error = dlerror()
                dlclose(handle)
                raise OSError(error.decode(_LOC_ENCODING, 'surrogateescape'))

            if not dladdr(addr, &info):
                error = dlerror()
                dlclose(handle)
                raise OSError(error.decode(_LOC_ENCODING, 'surrogateescape'))

            if info.dli_fname != NULL:
                dylib_path = info.dli_fname.decode(_FS_ENCODING,
                                                   'surrogateescape')

                # if libgap is in GAP_ROOT/.libs/
                gap_root = os.path.dirname(os.path.dirname(dylib_path))
                # On conda and sage (and maybe some other distros) we are in
                # <prefix>/ and gap is in share/gap
                # TODO: Add some other paths to try here as we find them
                for pth in [('.',), ('share', 'gap')]:
                    gap_root = os.path.join(gap_root, *pth)
                    if os.path.isfile(os.path.join(gap_root, 'lib', 'init.g')):
                        break
                else:
                    gap_root = None

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

    return os.path.normpath(gap_root)


cpdef get_gap_memory_pool_size(unit='m'):
    """
    Get the gap memory pool size for new GAP processes.

    Examples
    --------

    >>> from gappy.utils import get_gap_memory_pool_size
    >>> get_gap_memory_pool_size()  # system-specific output
    '...m'
    """
    allowed_units = ('k', 'm', 'g')
    unit = unit.lower()

    if unit not in allowed_units:
        raise ValueError(f'unit must be one of {", ".join(allowed_units)}')

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    vmax = virtual_memory_limit()

    suggested_size = max(swap.free // 10, mem.available // 50)
    # Don't eat all address space if the user set ulimit -v
    suggested_size = min(suggested_size, vmax // 10)
    # ~220MB is the minimum for long doctests
    suggested_size = max(suggested_size, 400 * 1024**2)
    unit_bytes = 1024**(allowed_units.index(unit) + 1)
    suggested_size //= unit_bytes
    return str(suggested_size) + unit


cpdef virtual_memory_limit():
    """
    Return the upper limit for virtual memory usage.

    This is the value set by ``ulimit -v`` at the command line or a
    practical limit if no limit is set. In any case, the value is
    bounded by ``sys.maxsize``.

    Returns
    -------

    int
        The virtual memory limit in bytes.

    Examples
    --------

    >>> from gappy.utils import virtual_memory_limit
    >>> virtual_memory_limit() > 0
    True
    >>> virtual_memory_limit() <= sys.maxsize
    True
    """
    try:
        vmax = resource.getrlimit(resource.RLIMIT_AS)[0]
    except resource.error:
        vmax = resource.RLIM_INFINITY
    if vmax == resource.RLIM_INFINITY:
        vmax = psutil.virtual_memory().total + psutil.swap_memory().total
    return min(vmax, sys.maxsize)
