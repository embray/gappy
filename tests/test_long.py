"""
Long tests for gappy.

These stress test the garbage collection inside GAP.
"""


from random import randint

import pytest

from gappy import gap


pytestmark = pytest.mark.long


def test_loop_1():
    gap.collect()
    for i in range(10000):
        G = gap.CyclicGroup(2)


def test_loop_2():
    G = gap.FreeGroup(2)
    a, b = G.GeneratorsOfGroup()
    for i in range(100):
        rel = gap([a**2, b**2, a*b*a*b])
        H = G / rel
        H1 = H.GeneratorsOfGroup()[0]
        n = H1.Order()
        assert n == 2

    for i in range(300000):
        n = gap.Order(H1)


def test_loop_3():
    G = gap.FreeGroup(2)
    a, b = G.GeneratorsOfGroup()
    for i in range(300000):
        lst = gap([])
        lst.Add(a ** 2)
        lst.Add(b ** 2)
        lst.Add(b * a)


def test_recursion_depth_overflow_on_error():
    """Regression test for https://github.com/embray/gappy/issues/12"""

    for i in range(0, 5000):
        rnd = [randint(-10, 10) for i in range(0, randint(0, 7))]
        # compute the sum in GAP
        _ = gap.Sum(rnd)
        try:
            gap.Sum(*rnd)
            pytest.fail(
                'This should have triggered a ValueError'
                'because Sum needs a list as argument'
            )
        except ValueError:
            pass

        # There's no reason this should ever go very high; 10 is a reasonable
        # upper-limit but in practice it never seems to go above 8 if this
        # is fixed properly
        assert gap.GetRecursionDepth() < 10
