Changelog
=========

v0.1.0a1 (unreleased)
---------------------

* Added LRU cache for functions defined with ``gap.gap_functions``,
  restoring some of the caching functionality from Sage's
  ``Gap.function_factory``.

* Fixed a bug when using functions defined with ``gap.gap_function`` as
  arguments to another GAP function before they have been called once.


v0.1.0a0 (2021-01-26)
---------------------

* Initial alpha release for testing against SageMath.
