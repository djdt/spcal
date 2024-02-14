import numpy as np

from spcal.nontarget import non_target_screen, screen_element


def test_screen_element():
    x = np.random.random(100)
    x[20::10] += 100.0
    x[10:12] += 100.0  # will be summed
    assert screen_element(x, mode="events") == 10
    assert screen_element(x, mode="detections") == 9


def test_non_target_screen():
    x = np.random.random((1000, 5))
    x[np.random.choice(1000, 10, replace=False), 1] += 100.0
    x[np.random.choice(1000, 20, replace=False), 2] += 100.0
    x[np.random.choice(1000, 40, replace=False), 3] += 100.0
    x[np.random.choice(1000, 80, replace=False), 4] += 100.0

    idx = non_target_screen(x, 100)
    assert np.all(idx == [1, 2, 3, 4])
    idx = non_target_screen(x, 15000)
    assert np.all(idx == [2, 3, 4])
    idx = non_target_screen(x, 30000)
    assert np.all(idx == [3, 4])
    idx = non_target_screen(x, 60000)
    assert np.all(idx == [4])
    idx = non_target_screen(x, 1000000)
    assert np.all(idx == [])
