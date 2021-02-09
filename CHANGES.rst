Changelog
=========

v0.1.0a3 (unreleased)
---------------------

Enhancements
^^^^^^^^^^^^

* Renamed the special method ``_gap_``, for converting arbitrary Python
  objects to GAP objects, to ``__gap__`` as inspired by the discussion at
  https://trac.sagemath.org/ticket/31297#comment:23

  * Likewise, the special method ``_gap_init_`` is now named
    ``__gap_eval__`` to emphasize that it returns a string to be passed
    to ``Gap.eval()``.  It still does not take any arguments.

* Added ``GapObj.python()`` method for converting a ``GapObj`` to its
  equivalent type if one exists (it does not always, but it does in the
  cases where there is an equivalent type built into Python).

  * ``GapList.python()`` and ``GapRecord.python()`` also recursively convert
    the values they contain to equivalent Python types if possible.

* New interface for registering converters to/from GAP object types:

  * ``Gap.register_converter`` is replaced with the ``Gap.convert_from``
    decorator.

  * The ``GapObj.convert_to`` decorator can be used to register new
    conversion methods on ``GapObj``, or specific subclasses thereof.

* Added some C-level utility methods on ``GapInteger`` to help convert to
  different integer types (C long ints and mpz_t, depending on the size of
  the int).  This helps with more efficient conversion to Sage Integers
  without having to pass through an intermediary Python ``int``.

* Implemented the ``__invert__`` and ``__neg__`` magic methods for
  ``GapObj``.

* Implemented a default ``__bool__`` for all ``GapObj`` which returns
  ``False`` if its value is equal to zero.

* Install the ``.pyx`` sources so that Cython tracebacks can work better.

Bug fixes
^^^^^^^^^

* When converting a ``GapRecord`` to a ``dict`` with ``dict(rec)`` the
  keys remain as ``GapString`` instead of ``str``.  This is more consistent
  with the fact that the values are not converted to Python equivalents.

* If an arbitrary GAP error occurs while looking up a global variable with
  ``Gap.__getattr__`` it is handled and re-raised as an ``AttributeError``.

* The ``Gap.__repr__`` method displays names of subclasses correctly.


v0.1.0a2 (2021-02-03)
---------------------

Bug fixes
^^^^^^^^^

* Made fixes for MacOS and Cygwin support.


v0.1.0a1 (2021-02-03)
---------------------

Enhancements
^^^^^^^^^^^^

* Added LRU cache for functions defined with ``gap.gap_functions``,
  restoring some of the caching functionality from Sage's
  ``Gap.function_factory``.

Bug fixes
^^^^^^^^^

* Fixed bug in multi-indexing of nested lists.

* Fixed minor formatting difference in the IndexError message when indexing
  single lists versus multi-indexing nested lists.

* Fixed a bug when using functions defined with ``gap.gap_function`` as
  arguments to another GAP function before they have been called once.


v0.1.0a0 (2021-01-26)
---------------------

* Initial alpha release for testing against SageMath.
