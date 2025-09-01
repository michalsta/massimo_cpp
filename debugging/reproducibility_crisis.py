# %pip install pandas
# %pip install mmappet
import massimo_cpp
import pandas as pd

import multiprocessing as mp
import numpy as np
import pickle
import shutil

from mmappet import open_dataset
from pathlib import Path
from typing import Iterable


DISCRETE_DISTRIBUTION = tuple[list[int], list[float]]


def write_clusters_to_mmappet(
    output_path: str | Path,
    frame_marginals: Iterable[DISCRETE_DISTRIBUTION],
    scan_marginals: Iterable[DISCRETE_DISTRIBUTION],
    tof_marginals: Iterable[DISCRETE_DISTRIBUTION],
    intensities: Iterable[int],
    minimal_reported_intensity: int = 9,
    precision: float = 0.999999,
    n_threads: int = mp.cpu_count() + 1,
    beta_bias: float = 0.1,
    seed: int | None = None,
    iso_backend: str = "layered",
    **kwargs,
) -> None:
    """Write clusters to the memmory mapped format (mmappet).

    The clusters are defined by marginal distributions that will be inputs for the IsoSpec algorithm that will also perform drawing from the multinomial distribution defined by the probabilities of the submitted marginals and the number of trials corresponding to the intensity, a.k.a. the number of ions to redistribute.
    This all works, for IsoSpec allows to draw efficiently from the multinomial distribution defined over probabilities that are themeselves products of multinomial probabilities. We abuse that last part and take multinomials with one experiment.

    Arguments:
        output_path (Path|str) Path to the folder with results.
        frame_marginals (tuple[list[int], list[float]]): Marginal distribution in the frame dimension: a list of frames and a list of corresponding probabilities.
        scan_marginals (tuple[list[int], list[float]]): Marginal distribution in the scan dimension: a list of scans and a list of corresponding probabilities.
        tof_marginals (tuple[list[int], list[float]]): Marginal distribution in the time of flight dimension: a list of tof pushes and a list of corresponding probabilities.
        intensities (Iterable[int]): How many ions to simulate per cluster.
        minimal_reported_intensity (int): Lower limit on the intensity of reported events: all events below will not be reported. THIS IS NOT AN OPTIMIZATION. THIS IS HOW INSTRUMENT WORKS. And btw, it is an optimization.
        precision (float): How to represent 1 in the numerics of isospec.
        n_threads (int): How many threads to use. Count as those that have to do computations simultaneously and one currently writing to disk.
        beta_bias (float): An integer telling how much to use beta distribution over binomial in the efficient simulation of the configurations.
    """
    problematic_inputs = []
    for frame_marginal, scan_marginal, tof_marginal, intensity in zip(
        frame_marginals, scan_marginals, tof_marginals, intensities
    ):
        problematic_inputs.append(
            massimo_cpp.ProblematicInput(
                intensity,
                minimal_reported_intensity,
                precision,
                *frame_marginal,
                *scan_marginal,
                *tof_marginal,
            )
        )
    massimo_cpp.Massimize(
        inputs=problematic_inputs,
        output_dir_path=str(output_path),
        n_threads=n_threads,
        beta_bias=beta_bias,
        seed=seed,
        iso_backend=iso_backend,
    )


with open("/tmp/repro.pkl", "rb") as f:
    precursor_frame_marginals, precursor_tof_marginals, ions = pickle.load(f)

minimal_intensity = 9
precision = 0.99

precursors_output_path1 = "/tmp/test1.mmappet"
precursors_output_path2 = "/tmp/test2.mmappet"

shutil.rmtree(precursors_output_path1, ignore_errors=True)
write_clusters_to_mmappet(
    frame_marginals=precursor_frame_marginals,
    scan_marginals=ions.scan_marginal,
    tof_marginals=precursor_tof_marginals,
    intensities=ions.intensity,
    minimal_reported_intensity=minimal_intensity,
    output_path=precursors_output_path1,
    seed=1,
)

shutil.rmtree(precursors_output_path2, ignore_errors=True)
write_clusters_to_mmappet(
    frame_marginals=precursor_frame_marginals,
    scan_marginals=ions.scan_marginal,
    tof_marginals=precursor_tof_marginals,
    intensities=ions.intensity,
    minimal_reported_intensity=minimal_intensity,
    output_path=precursors_output_path2,
    seed=1,
)

df = open_dataset(precursors_output_path1)
df2 = open_dataset(precursors_output_path2)
df
df2
