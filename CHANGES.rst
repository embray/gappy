Changelog
=========

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
