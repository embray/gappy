# distutils: libraries = gap gmp m
###############################################################################
#       Copyright (C) 2009, William Stein <wstein@gmail.com>
#       Copyright (C) 2012, Volker Braun <vbraun.name@gmail.com>
#       Copyright (C) 2021 E. Madison Bray <embray@lri.fr>
#
#   Distributed under the terms of the GNU General Public License (GPL)
#   as published by the Free Software Foundation; either version 2 of
#   the License, or (at your option) any later version.
#                   http://www.gnu.org/licenses/
###############################################################################

from libc.stdint cimport uintptr_t, uint8_t, uint16_t, uint32_t, uint64_t


cdef extern from "gap/libgap-api.h" nogil:
    r"""
    /* The recursion_depth stuff is a temporary work-around to
     * https://github.com/embray/gappy/issues/12 and necessitates some
     * use of GAP internals; it can go away after GAP 4.12 (we should
     * maybe do a GAP version check before enabling this hack)
     */
    #include "gap/funcs.h"
    static volatile Int _gappy_recursion_depth = 0;
    #define sig_GAP_Enter() { \
        int _gappy_ok = GAP_Enter(); \
        if (_gappy_ok) { \
            if (_gappy_recursion_depth == 0) { \
                _gappy_recursion_depth = GetRecursionDepth(); \
            } \
        } else { \
            sig_error(); \
        } \
    }
    #define sig_GAP_Leave() { \
        if (_gappy_recursion_depth != 0) { \
            SetRecursionDepth(_gappy_recursion_depth); \
            _gappy_recursion_depth = 0; \
        } \
        GAP_Leave(); \
    }
    """

    # Basic types
    ctypedef char Char
    ctypedef int Int
    ctypedef uintptr_t UInt
    ctypedef uint8_t  UInt1
    ctypedef uint16_t UInt2
    ctypedef uint32_t UInt4
    ctypedef uint64_t UInt8
    ctypedef void* Obj

    # Stack management
    cdef void GAP_EnterStack()
    cdef void GAP_LeaveStack()
    cdef int GAP_Enter() except 0
    cdef void sig_GAP_Enter()
    cdef void GAP_Leave()
    cdef void sig_GAP_Leave()
    cdef int GAP_Error_Setjmp() except 0

    # Initialization
    ctypedef void (*GAP_CallbackFunc)()
    void GAP_Initialize(int argc, char ** argv,
            GAP_CallbackFunc markBagsCallback, GAP_CallbackFunc errorCallback,
            int handleSignals)

    # Arithmetic and operators
    Obj GAP_SUM(Obj, Obj)
    Obj GAP_DIFF(Obj, Obj)
    Obj GAP_PROD(Obj, Obj)
    Obj GAP_QUO(Obj, Obj)
    Obj GAP_POW(Obj, Obj)
    Obj GAP_MOD(Obj, Obj)
    int GAP_EQ(Obj opL, Obj opR)
    int GAP_LT(Obj opL, Obj opR)
    int GAP_IN(Obj, Obj)

    # Booleans
    cdef Obj GAP_True
    cdef Obj GAP_False

    # Evaluation
    Obj GAP_EvalString(const char *) except *
    Obj GAP_EvalStringNoExcept "GAP_EvalString"(const char *)

    # Global variables
    void GAP_AssignGlobalVariable(const char *, Obj)
    int GAP_CanAssignGlobalVariable(const char *)
    Obj GAP_ValueGlobalVariable(const char *)

    # Calls
    Obj GAP_CallFuncArray(Obj, UInt, Obj *)
    Obj GAP_CallFuncList(Obj, Obj)

    # Ints
    cdef int GAP_IsInt(Obj)
    cdef int GAP_IsSmallInt(Obj)
    cdef Obj GAP_MakeObjInt(UInt *, Int)
    cdef Int GAP_SizeInt(Obj)
    cdef UInt *GAP_AddrInt(Obj)

    # Floats
    cdef Obj GAP_NewMacFloat(double)
    double GAP_ValueMacFloat(Obj)

    # Strings
    cdef char *GAP_CSTR_STRING(Obj)
    cdef int GAP_IsString(Obj)
    cdef UInt GAP_LenString(Obj)
    cdef Obj GAP_MakeString(const char *)

    # Lists
    void GAP_AssList(Obj, UInt, Obj val)
    Obj GAP_ElmList(Obj, UInt)
    UInt GAP_LenList(Obj)
    int GAP_IsList(Obj)
    Obj GAP_NewPlist(Int)

    # Records
    void GAP_AssRecord(Obj, Obj, Obj)
    int GAP_IsRecord(Obj)
    Obj GAP_ElmRecord(Obj, Obj)
    Obj GAP_NewPrecord(Int)


cdef extern from "gap/gasman.h" nogil:
    """
    #define GAP_CollectBags(full) CollectBags(0, full)
    """
    void GAP_MarkBag(Obj bag)
    UInt GAP_CollectBags(UInt full)


cdef extern from "gap/io.h" nogil:
    """
    TypOutputFile output = {0};
    inline UInt GAP_OpenOutputStream(Obj stream) {
        return OpenOutputStream(&output, stream);
    }
    inline UInt GAP_CloseOutput( ) {
        return CloseOutput(&output);
    }
    """
    UInt GAP_OpenOutputStream(Obj stream)
    UInt GAP_CloseOutput()


# TODO: Replace this with a GAP_MakeStringWithLen from the public API;
# see https://github.com/gap-system/gap/issues/4211
cdef extern from "gap/stringobj.h" nogil:
    """
    inline Obj GAP_MakeStringWithLen(const char *s, size_t len) {
        Obj ret;
        C_NEW_STRING(ret, len, s);
        return ret;
    }
    """
    Obj GAP_MakeStringWithLen(char *, size_t)
