"""
Long tests for gappy.

These stress test the garbage collection inside GAP.
"""


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
