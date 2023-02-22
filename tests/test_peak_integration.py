from pathlib import Path

import numpy as np

from spcal import detection, poisson


def test_noise_level():
    """Tests detections / integrations staty the same when noise levels rise.
    Going further beyond the limit of crit will start to integrate noise.
    """
    data = np.load(Path(__file__).parent.joinpath("data/tofwerk_auag.npz"))
    for name in ["107Ag", "197Au"]:
        x = data[name]
        ub = np.mean(x)
        yc, _ = poisson.formula_c(ub, alpha=0.001)
        detections, _, _ = detection.accumulate_detections(
            x, ub, yc + ub, integrate=True
        )

        truth = detections.size
        truth_mean = detections.mean()

        np.random.seed(142893)

        for lam in np.linspace(ub, yc, 5, endpoint=True):
            noise = np.random.poisson(lam=lam, size=x.size)

            noise_x = x + noise

            noise_ub = np.mean(noise_x)
            noise_yc, _ = poisson.formula_c(ub, alpha=0.001)
            detections, _, _ = detection.accumulate_detections(
                noise_x, noise_ub, noise_yc + noise_ub, integrate=True
            )

            # Less than 1% change in number of detections
            assert abs(truth - detections.size) / truth < 0.01
            # Less than 5% change in mean detected area
            assert abs(truth_mean - detections.mean()) / truth_mean < 0.05
