"""Additional pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def inject_globals(doctest_namespace):
    """
    Make certain variables available globally to all doctests; in particular
    the global `~gappy.core.Gap` instance ``gap`` which is used in most tests.
    """

    from gappy import gap
    doctest_namespace['gap'] = gap
