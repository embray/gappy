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

from .gap_includes cimport Obj
from .gmp cimport gmp_randstate_t

############################################################################
### Hooking into the GAP memory management #################################
############################################################################

cdef class ObjWrapper(object):
    cdef Obj value

cdef ObjWrapper wrap_obj(Obj obj)

# returns the refcount dictionary for debugging purposes
cpdef get_owned_objects()

# Reference count GAP objects that you want to prevent from being
# garbage collected
cdef void reference_obj(Obj obj)
cdef void dereference_obj(Obj obj)


############################################################################
### Initialization of GAP ##################################################
############################################################################


cdef class Gap:
    cdef dict _init_kwargs
    cdef readonly tuple supported_builtins
    cdef readonly dict _converter_registry
    cdef gmp_randstate_t _gmp_state
    cpdef initialize(self)
    cpdef _from_gap_init(self, x)
    cpdef eval(self, gap_command)
    cpdef get_global(self, variable)
