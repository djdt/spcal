"""These tests use data generated in SPTool 3.2.1
https://doi.org/10.1039/D3JA00292F
"""

from pathlib import Path

import numpy as np

from spcal.cluster import cluster_information, prepare_results_for_clustering
from spcal.datafile import SPCalTextDataFile
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod


def test_simulated_backgrounds(test_data_path: Path):
    method = SPCalProcessingMethod()
    method.limit_options.limit_method = "poisson"
    method.limit_options.poisson_kws["alpha"] = 1e-7
    method.limit_options.max_iterations = 100
    method.processing_options.prominence_required = 0.5
    method.processing_options.points_required = 1

    path = test_data_path.joinpath("simulations/sptool_background_data.npz")
    data = np.load(path)

    times = np.arange(5000) * 0.0001
    au = ISOTOPE_TABLE[("Au", 197)]

    for bg in [0, 1, 10, 100]:
        signals = data[f"mean_1000_bg_{bg}"].astype([("sim", np.float32)])
        df = SPCalTextDataFile(path, signals, times, {au: "sim"})

        result = method.processDataFile(df, [au])[au]

        assert np.isclose(
            result.background, bg, atol=0.01, rtol=0.05
        )  # 5 % off, or 0.01

        assert np.isclose(np.mean(result.detections), 1000.0, rtol=0.05)  # 5 % off
        assert np.isclose(np.median(result.detections), 1000.0, rtol=0.01)  # 1 % off


def test_simulated_compositions(test_data_path: Path):
    method = SPCalProcessingMethod()
    method.limit_options.limit_method = "poisson"
    method.limit_options.poisson_kws["alpha"] = 1e-7
    method.limit_options.max_iterations = 100
    method.processing_options.prominence_required = 0.5
    method.processing_options.points_required = 1
    method.processing_options.cluster_distance = 0.05

    path = test_data_path.joinpath("simulations/sptool_composition_data.npz")
    data = np.load(path)["ru"]
    times = np.arange(5000) * 0.0001

    df = SPCalTextDataFile(
        path,
        data,
        times,
        {SPCalIsotope.fromString(name): name for name in data.dtype.names},
    )
    df.selected_isotopes = df.isotopes

    results = method.processDataFile(df)
    method.filterResults(results)

    npeaks = next(iter(results.values())).number_peak_indicies
    clusters = method.processClusters(results)

    X, _ = prepare_results_for_clustering(list(results.values()), npeaks, "signal")
    means, _, counts = cluster_information(X, clusters)

    natural_ru = np.array(
        [ISOTOPE_TABLE[("Ru", x)].composition for x in [96, 98, 99, 100, 101, 102, 104]]
    )

    assert counts[0] > np.sum(counts) - 10  # mostly a single cluster
    assert np.allclose(
        means[0], natural_ru, atol=0.01
    )  # all with 0.01 of natural abundance
