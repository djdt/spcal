import numpy as np
from numpy.lib import stride_tricks

from spcal.result import SPCalResult



# def test_results_from_mass_response():
#     detections = np.array([10.0, 20.0, 30.0])
#     r = SPCalResult(np.arra([]), detections=detections, labels=np.array([]), )
#     results = calc.results_from_mass_response(
#         detections,
#         background=1.0,
#         density=0.01,
#         lod=5.0,
#         mass_fraction=0.5,
#         mass_response=1e-3,
#     )
#     assert all(
#         x in results
#         for x in ["masses", "sizes", "background_size", "lod", "lod_mass", "lod_size"]
#     )
#     assert np.all(results["masses"] == detections * 1e-3 / 0.5)
#     assert results["lod_mass"] == 5.0 * 1e-3 / 0.5
#     # Rest are calculated as per particle module


# def test_results_from_nebulisation_efficieny():
#     detections = np.array([10.0, 20.0, 30.0])
#     results = calc.results_from_nebulisation_efficiency(
#         detections,
#         background=1.0,
#         density=0.01,
#         lod=5.0,
#         dwelltime=1e-3,
#         efficiency=0.05,
#         uptake=0.2,
#         time=60.0,
#         response=100.0,
#         mass_fraction=0.5,
#     )
#     assert all(
#         x in results
#         for x in ["masses", "sizes", "background_size", "lod", "lod_mass", "lod_size"]
#     )
#     assert all(
#         x in results
#         for x in ["concentration", "number_concentration", "background_concentration"]
#     )
#     assert results["background_concentration"] == 1.0 / 100.0
#     # Rest are calculated as per particle module
