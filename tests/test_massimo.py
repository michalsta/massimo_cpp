#!/usr/bin/env python

import massimo_cpp
from mmapped_df import open_dataset
import numpy as np


def get_probs_arr(n):
    a = np.random.rand(n)
    a /= a.sum()
    return a


def test_1():
    PI = massimo_cpp.ProblematicInput(
        100000,
        27,
        0.99,
        [1,2],
        [0.9, 0.1],
        [34, 35, 36, 37, 38],
        [0.1, 0.2, 0.3, 0.2, 0.2],
        [1,5,6,7],
        [0.1, 0.2, 0.3, 0.4]
    )

    massimo_cpp.Massimize([PI], 1, r"0.bin")

    ds = open_dataset(r".")
    print(ds)
    assert sum(ds.data) == 100_000

def test_2():
    PI = massimo_cpp.ProblematicInput(
        100000,
        27,
        0.99,
        [1,2],
        [0.9, 0.1],
        [34, 35, 36, 37, 38],
        [0.1, 0.2, 0.3, 0.2, 0.2],
        [1,5,6,7],
        [0.1, 0.2, 0.3, 0.4]
    )

    massimo_cpp.Massimize([PI]*1000, 20, r"0.bin")

    ds = open_dataset(r".")
    print(ds)
    assert sum(ds.data) == 100000000