"""Cython definitions for miscellaneous bits from GMP."""


from .gap_includes cimport UInt


cdef extern from "<gmp.h>":
    ctypedef struct __mpz_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]

    ctypedef struct gmp_randstate_t:
        pass

    ctypedef unsigned long mp_bitcnt_t
    # GAP ensures at compile time that sizeof(mp_limb_t) == sizeof(UInt)
    ctypedef UInt mp_limb_t

    void mpz_init(mpz_t)
    void mpz_clear(mpz_t)
    void mpz_neg(mpz_t, const mpz_t)
    void mpz_import(mpz_t, size_t, int, int, int, size_t, void *)
    void *mpz_export(void *, size_t *, int, size_t, int, size_t, const mpz_t)
    size_t mpz_size(const mpz_t)
    size_t mpz_sizeinbase(const mpz_t, int)
    const mp_limb_t* mpz_limbs_read(const mpz_t)

    void gmp_randinit_default(gmp_randstate_t)
    void mpz_rrandomb(mpz_t, gmp_randstate_t, mp_bitcnt_t)
