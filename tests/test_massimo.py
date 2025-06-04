#!/usr/bin/env python

import massimo_cpp
from mmapped_df import open_dataset
import numpy as np
import shutil

def get_probs_arr(n):
    a = np.random.rand(n).astype(np.double)
    a /= a.sum()
    a *= 1.1
    return a.tolist()


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
    massimo_cpp.Massimize([PI], r"test1.mmappet", 1)

    ds = open_dataset(r"test1.mmappet")
    print(ds)
    assert sum(ds.intensity) == 100_000

def test_2():
    PI = massimo_cpp.ProblematicInput(
        100000,
        27,
        0.99,
        np.array([1,2]),
        np.array([0.9, 0.1]),
        np.array([34, 35, 36, 37, 38]),
        np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        np.array([1,5,6,7]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )

    shutil.rmtree(r"test2.mmappet", ignore_errors=True)
    massimo_cpp.Massimize([PI]*1000, r"test2.mmappet", 20)

    ds = open_dataset(r"test2.mmappet")
    print(ds)
    assert sum(ds.intensity) == 100000000

def test_3():
    PI = massimo_cpp.ProblematicInput(
        100000,
        27,
        0.99,
        np.arange(50, dtype=np.uint64).tolist(),
        get_probs_arr(50),
        np.arange(100, dtype=np.uint64).tolist(),
        get_probs_arr(100),
        np.arange(50, dtype=np.uint64).tolist(),
        get_probs_arr(50),
    )
    #print(PI.to_cpp_string())
    shutil.rmtree(r"test3.mmappet", ignore_errors=True)
    massimo_cpp.Massimize([PI]*1, r"test3.mmappet", 13)
    ds = open_dataset(r"test3.mmappet")
    print(ds)

if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    print("All tests passed.")