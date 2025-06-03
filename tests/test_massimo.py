#!/usr/bin/env python

import massimo_cpp
from mmapped_df import open_dataset
import numpy as np
import shutil

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

    shutil.rmtree(r"test1.mappet", ignore_errors=True)
    massimo_cpp.Massimize([PI], 1, r"test1.mmappet")

    ds = open_dataset(r"test1.mmappet")
    print(ds)
    assert sum(ds.intensity) == 100_000

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

    shutil.rmtree(r"test2.mmappet", ignore_errors=True)
    massimo_cpp.Massimize([PI]*1000, 20, r"test2.mmappet")

    ds = open_dataset(r"test2.mmappet")
    print(ds)
    assert sum(ds.intensity) == 100000000

if __name__ == "__main__":
    test_1()
    test_2()
    print("All tests passed.")