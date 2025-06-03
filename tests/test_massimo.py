#!/usr/bin/env python

import massimo_cpp
massimo_cpp.hello()
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

from mmapped_df import open_dataset
ds = open_dataset(r".")
print(ds)