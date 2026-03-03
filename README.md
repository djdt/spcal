# SPCal

SPCal in an spICP-MS calculator with an interactive GUI.

Both quadrupole and ToF data is supported, with native import from Nu Instruments and TOFWERK ICPs.

<img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_2_0_3_main_window.png" width=600>

## Processing

SPCal has basic data processing and export functions including:
- histograms and distribution fitting
- particle composition by heirachical clustering
- scatter plots
- particle filtering by signal / mass / size
- per mass SIA recovery from ionic data


<img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_0_9_13_histogram.png" width=300><img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_0_9_13_composition.png" width=300>

## Installation

Windows executables are available for each [release](https://github.com/djdt/spcal/releases).

To install via pip first clone the repository then install as a local package.
Installion requires compilation of a C++ Extension (linking [TBB](https://github.com/uxlfoundation/oneTBB)), see [Extending Python with C or C++](https://docs.python.org/3/extending/extending.html).

```bash
git clone https://github.com/djdt/spcal
cd spcal
pip install -e .
```

The program can then be run from the command line. See the help for extended usage.

```bash
spcal --help
```

## Documentation

Documentation on usage, examples and a programming reference can be found at https://spcal.readthedocs.io.

## Publications

* [Lockwood, T. E.; Gonzalez de Vega, R.; Clases, D. An Interactive Python-Based Data Processing Platform for Single Particle and Single Cell ICP-MS. Journal of Analytical Atomic Spectrometry 2021, 36 (11), 2536–2544.](https://doi.org/10.1039/D1JA00297J)
* [Lockwood, T. E.; Schlatt, L.; Clases, D. SPCal – an Open Source, Easy-to-Use Processing Platform for ICP-TOFMS-Based Single Event Data. J. Anal. At. Spectrom. 2025, 40, 130-136](https://doi.org/10.1039/D4JA00241E)
* [Lockwood, T. E.; Gonzalez de Vega, R.; Schaltt, L.; Clases, D. Accurate thresholding using a compound-Poisson-lognormal lookup table and parameters recovered from standard single particle ICP-TOFMS data. J. Anal. At. Spectrom. 2025, 40, 2633-2640](https://doi.org/10.1039/D5JA00230C)
