# ****************************************************************************
#       Copyright (C) 2006 William Stein <wstein@gmail.com>
#       Copyright (C) 2021 E. Madison Bray <embray@lri.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************


def get_gap_memory_pool_size():
    """
    Get the gap memory pool size for new GAP processes.

    EXAMPLES::

        sage: from sage.interfaces.gap import get_gap_memory_pool_size
        sage: get_gap_memory_pool_size()   # random output
        1534059315
    """
    global gap_memory_pool_size
    if gap_memory_pool_size is not None:
        return gap_memory_pool_size

    import psutil
    from sage.misc.getusage import virtual_memory_limit
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    vmax = virtual_memory_limit()

    suggested_size = max(swap.free // 10, mem.available // 50)
    # Don't eat all address space if the user set ulimit -v
    suggested_size = min(suggested_size, vmax // 10)
    # ~220MB is the minimum for long doctests
    suggested_size = max(suggested_size, 400 * 1024**2)
    return suggested_size


def virtual_memory_limit():
    """
    Return the upper limit for virtual memory usage.

    This is the value set by ``ulimit -v`` at the command line or a
    practical limit if no limit is set. In any case, the value is
    bounded by ``sys.maxsize``.

    OUTPUT:

    Integer. The virtual memory limit in bytes.

    EXAMPLES::

        sage: from sage.misc.getusage import virtual_memory_limit
        sage: virtual_memory_limit() > 0
        True
        sage: virtual_memory_limit() <= sys.maxsize
        True
    """
    import resource
    try:
        vmax = resource.getrlimit(resource.RLIMIT_AS)[0]
    except resource.error:
        vmax = resource.RLIM_INFINITY
    if vmax == resource.RLIM_INFINITY:
        import psutil
        vmax = psutil.virtual_memory().total + psutil.swap_memory().total
    return min(vmax, sys.maxsize)
