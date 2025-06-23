Background and Theory
=====================

Single-particle ICP-MS is a technique to analyse particulate matter in aqueous samples.
It is essentially the same as any time-resolved ICP-MS measurement, but with a significantly shorter acquisition or :term:`dwelltime`.
The rapid collection of data (typically sub millisecond) allows the resolution of individual particles as they are nebulised, atomised and extracted in discrete ion clouds.
For a comprehensive review of spICP-MS as a nano-particle analysis technique see Laborda, Bolea and Jiménez-Lamana [1]_.

Thresholds
----------

One of the major difficulties with spICP-MS is establishing theoretically sound detection thresholds.
These pages discuss the use and implementation of Gaussian, Poisson and Compound-Poisson thresholds in SPCal.

.. toctree::
   thresholds

Parameter Recovery
------------------

The methods used in SPCal for thresholding ICP-TOF data require knowledge of the underlying distribution.
Here we explain how we can recover this knowledge from existing data and use it to accurately threhold.

.. toctree::
   extraction

Publications
------------

A previous, quadruple only, version of SPCal was published and is available here:

* Lockwood, T. E.; Gonzalez de Vega, R.; Clases, D. An Interactive Python-Based Data Processing Platform for Single Particle and Single Cell ICP-MS. Journal of Analytical Atomic Spectrometry 2021, 36 (11), 2536–2544. `<https://doi.org/10.1039/D1JA00297J>`_.

Several other publications discussing aspects of the new ToF version are also available:

* Gonzalez De Vega, R.; Lockwood, T. E.; Paton, L.; Schlatt, L.; Clases, D. Non-Target Analysis and Characterisation of Nanoparticles in Spirits via Single Particle ICP-TOF-MS. Journal of Analytical Atomic Spectrometry 2023, 10.1039.D3JA00253E. `<https://doi.org/10.1039/D3JA00253E>`_.

* Lockwood, T. E.; Gonzalez De Vega, R.; Du, Z.; Schlatt, L.; Xu, X.; Clases, D. Strategies to Enhance Figures of Merit in ICP-ToF-MS. Journal of Analytical Atomic Spectrometry 2024, 39 (1), 227–234. `<https://doi.org/10.1039/D3JA00288H>`_.

.. [1] Laborda, F.; Bolea, E.; Jiménez-Lamana, J. Single Particle Inductively Coupled Plasma Mass Spectrometry: A Powerful Tool for Nanoanalysis. Anal. Chem. 2014, 86 (5), 2270–2278. https://doi.org/10.1021/ac402980q.
