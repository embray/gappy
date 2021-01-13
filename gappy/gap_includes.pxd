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

cdef extern from "gap/system.h" nogil:
    ctypedef char Char
    ctypedef int Int
    ctypedef uintptr_t UInt
    ctypedef uint8_t  UInt1
    ctypedef uint16_t UInt2
    ctypedef uint32_t UInt4
    ctypedef uint64_t UInt8
    ctypedef void* Obj


cdef extern from "gap/ariths.h" nogil:
    Obj SUM(Obj, Obj)
    Obj DIFF(Obj, Obj)
    Obj PROD(Obj, Obj)
    Obj QUO(Obj, Obj)
    Obj POW(Obj, Obj)
    Obj MOD(Obj, Obj)
    bint EQ(Obj opL, Obj opR)
    bint LT(Obj opL, Obj opR)


cdef extern from "gap/bool.h" nogil:
    cdef Obj GAP_True "True"
    cdef Obj GAP_False "False"


cdef extern from "gap/calls.h" nogil:
    bint IS_FUNC(Obj)


cdef extern from "gap/gasman.h" nogil:
    Obj NewBag "NewBag"(UInt type, UInt size)
    void MarkBag(Obj bag)
    UInt CollectBags(UInt size, UInt full)


cdef extern from "gap/gasman_intern.h" nogil:
    void CallbackForAllBags(void (*func)(Obj))


cdef extern from "gap/gvars.h" nogil:
    UInt GVarName "GVarName"(char* name)
    void AssGVar "AssGVar"(UInt gvar, Obj val)


cdef extern from "gap/io.h" nogil:
    UInt OpenOutputStream(Obj stream)
    UInt CloseOutput()


cdef extern from "gap/libgap-api.h" nogil:
    """
    #define sig_GAP_Enter()  {int t = GAP_Enter(); if (!t) sig_error();}
    """
    ctypedef void (*GAP_CallbackFunc)()
    void GAP_Initialize(int argc, char ** argv,
            GAP_CallbackFunc markBagsCallback, GAP_CallbackFunc errorCallback,
            int handleSignals)
    Obj GAP_EvalString(const char *) except *
    Obj GAP_EvalStringNoExcept "GAP_EvalString"(const char *)
    Obj GAP_ValueGlobalVariable(const char *)
    cdef void GAP_EnterStack()
    cdef void GAP_LeaveStack()
    cdef int GAP_Enter() except 0
    cdef void sig_GAP_Enter()
    cdef void GAP_Leave()
    cdef int GAP_Error_Setjmp() except 0

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


cdef extern from "gap/macfloat.h" nogil:
    double VAL_MACFLOAT(Obj obj)


cdef extern from "gap/objects.h" nogil:
    bint IS_MUTABLE_OBJ(Obj obj)
    Obj SHALLOW_COPY_OBJ(Obj obj)
    Obj CopyObj(Obj obj, int mut)

    UInt SIZE_OBJ(Obj obj)
    UInt TNUM_OBJ(Obj obj)
    char* TNAM_OBJ(Obj obj)

    cdef enum TNUM:
        T_INT
        T_INTPOS
        T_INTNEG
        T_RAT
        T_CYC
        T_FFE
        T_MACFLOAT
        T_PERM2
        T_PERM4
        T_TRANS2
        T_TRANS4
        T_PPERM2
        T_PPERM4
        T_BOOL
        T_CHAR
        T_FUNCTION
        T_PLIST
        T_PLIST_CYC
        T_BLIST
        T_STRING
        T_COMOBJ
        T_POSOBJ
        T_DATOBJ
        T_WPOBJ


cdef extern from "gap/permutat.h" nogil:
    UInt DEG_PERM2(Obj)
    UInt DEG_PERM4(Obj)
    Obj NEW_PERM2(UInt)
    Obj NEW_PERM4(UInt)
    UInt2* ADDR_PERM2(Obj)
    UInt4* ADDR_PERM4(Obj)
    const UInt2* CONST_ADDR_PERM2(Obj)
    const UInt4* CONST_ADDR_PERM4(Obj)


cdef extern from "gap/stringobj.h" nogil:
    Obj GAP_MakeStringWithLen "MakeStringWithLen" (char *, size_t)
