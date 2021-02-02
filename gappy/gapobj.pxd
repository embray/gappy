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

from .gap_includes cimport Obj, UInt

cdef Obj make_gap_list(parent, lst) except NULL
cdef Obj make_gap_record(parent, dct) except NULL
cdef Obj make_gap_integer(x) except NULL
cdef Obj make_gap_float(x) except NULL
cdef Obj make_gap_string(s) except NULL
cdef GapObj make_any_gap_obj(parent, Obj obj)

cdef GapObj make_GapObj(parent, Obj obj)
cdef GapList make_GapList(parent, Obj obj)
cdef GapRecord make_GapRecord(parent, Obj obj)
cdef GapInteger make_GapInteger(parent, Obj obj)
cdef GapFloat make_GapFloat(parent, Obj obj)
cdef GapRational make_GapRational(parent, Obj obj)
cdef GapString make_GapString(parent, Obj obj)
cdef GapBoolean make_GapBoolean(parent, Obj obj)
cdef GapFunction make_GapFunction(parent, Obj obj)
cdef _GapLazyFunction make_GapLazyFunction(parent, str name, str doc,
                                           str source)
cdef GapPermutation make_GapPermutation(parent, Obj obj)

cdef void capture_stdout(Obj, Obj, Obj)
cdef void gap_obj_str(Obj, Obj)
cdef void gap_obj_repr(Obj, Obj)


cdef class GapObj:
    # the instance of the Gap interpreter class; currently for compatibility
    # with Sage's Element class though not clear yet if it will make entire
    # sense to keep.
    cdef object _parent

    # the pointer to the GAP object (memory managed by GASMAN)
    cdef Obj value

    # comparison
    cdef bint _compare_by_id
    cdef bint _compare_equal(self, GapObj other) except -2
    cdef bint _compare_less(self, GapObj other) except -2
    cpdef _set_compare_by_id(self)
    cpdef _assert_compare_by_id(self)

    cdef _initialize(self, parent, Obj obj)
    cpdef is_bool(self)
    cpdef _add_(self, other)
    cpdef _div_(self, other)
    cpdef _sub_(self, other)
    cpdef _mul_(self, other)
    cpdef _mod_(self, other)
    cpdef _pow_(self, other)
    cpdef _richcmp_(self, other, int op)

    cpdef GapObj deepcopy(self, bint mut)

cdef class GapInteger(GapObj):
    cpdef is_C_int(self)

cdef class GapFloat(GapObj):
    pass

cdef class GapRational(GapObj):
    pass

cdef class GapIntegerMod(GapObj):
    cpdef GapInteger lift(self)

cdef class GapFiniteField(GapObj):
    cpdef GapInteger lift(self)

cdef class GapCyclotomic(GapObj):
    pass

cdef class GapRing(GapObj):
    pass

cdef class GapString(GapObj):
    pass

cdef class GapBoolean(GapObj):
    pass

cdef class GapFunction(GapObj):
    cdef str name
    cdef str doc

cdef class _GapLazyFunction(GapFunction):
    cdef str source
    cdef resolve(self)

cdef class GapMethodProxy(GapFunction):
    cdef GapFunction func
    cdef object self

cdef class GapList(GapObj):
    pass

cdef class GapRecord(GapObj):
    cdef GapList _names(self)
    cdef GapObj _getitem(self, GapString name)

cdef class GapPermutation(GapObj):
    pass
