# SPCal

SPCal in an spICP-MS calculator with an interactive GUI.

Both quadrupole and ToF data is supported, with native import from Nu Instruments and TOFWERK ICPs.

<img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_0_9_13_sample_window.png" width=600>

## Processing

SPCal has basic data processing and export functions including:
- histograms and distribution fitting
- particle composition by heirachical clustering
- scatter plots and PCAs
- particle filtering by signal / mass / size


<img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_0_9_13_histogram.png" width=300><img src="https://github.com/djdt/djdt.github.io/blob/main/img/spcal_0_9_13_composition.png" width=300>

## Installation

Windows executables are available for each [release](https://github.com/djdt/spcal/releases).

To install via pip first clone the repository then install as a local package.

```bash
git clone https://github.com/djdt/spcal
cd spcal
pip install -e .
```

The program can then be run from the command line. See the help for extended usage.

```bash
spcal --help
```

## Publications

* [Lockwood, T. E.; Gonzalez de Vega, R.; Clases, D. An Interactive Python-Based Data Processing Platform for Single Particle and Single Cell ICP-MS. Journal of Analytical Atomic Spectrometry 2021, 36 (11), 2536–2544.](https://doi.org/10.1039/D1JA00297J)
