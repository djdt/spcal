Calibration
===========

One of the advantages of spICP-MS is its ability to calculate the size and mass distribution of particles.
This is performed by calibrating particle signals (in counts) to mass using the instrument :term:`ionic response`, and then to size using a known particle :term:`density`.

The :term:`transport efficiency` is the fraction of sample that makes it through to detection and must also be determined before calibration can occour.
With the exception of total-consumption nebulisers (100% efficiency) it is typically 0.02 - 0.1 (2 - 10%).
The :term:`transport efficiency` (:math:`\eta`) can be entered manually if known (see Pace et al. [1]_ for examples), or calulated based on the reponse of a well characterised reference particle.

A full description of the calibration methods used in SPCal is avaiable in a previous publication [2]_ .

Reference Particle
------------------

.. _calibrate reference tab:
.. figure:: images/tutorial_cal_reference.png
   :align: center

   The Reference tab is used to calculate the transport efficiency.
   This is required to calibrate data from signal into mass and size.

To use a reference particle select *Reference Particle* as the :term:`transport efficiency` option in the **Options Tab**.
This enables the **Reference Tab**, where data for the reference particle can be loaded via drag-and-drop or **File -> Open Reference File**.
See :ref:`Data Import` for details on importing data.

To correctly calibrate, the particle :term:`diameter`, :term:`density` and :term:`ionic response` must be enetered in the **Reference Tab** and the instrument :term:`uptake` in the **Options Tab**.
Ideally a particle of a single element is used, if one containing multiple is used then the :term:`mass fraction` of the measured element must be entered.
If the concentration of the reference particle solution is known then the accuracy of the calculation will be greater.
Once all parameters are input, the calculated efficiency is shown in the **Reference Tab** outputs section.

The :term:`transport efficiency` is usually assumed to be idependent of mass and a single element can be used to calibrate the entire mass range.
Selecting the *Calibrate for all elements* will use the currently selected element in the **Reference Tab** to determine the :term:`transport efficiency`.
If not selected, each element will *only calibrate data with the same element name* in the **Sample Tab**.


Mass Response
-------------

Limited calibration can also occur with the :term:`transport efficiency` by determining the :term:`mass response` from a reference particle.
After selecting *Mass Response* as the :term:`transport efficiency` option in the **Options Tab** the calibration proceeds as above in the `Reference Particle`_ section.
Using the :term:`mass response` eliminates the need for instrument :term:`uptake` and :term:`ionic response` but can only calibrate signals into masses.


.. [1] Pace, H. E.; Rogers, N. J.; Jarolimek, C.; Coleman, V. A.; Higgins, C. P.; Ranville, J. F. Determining Transport Efficiency for the Purpose of Counting and Sizing Nanoparticles via Single Particle Inductively Coupled Plasma Mass Spectrometry. Anal. Chem. 2011, 83 (24), 9361–9369. https://doi.org/10.1021/ac201952t.

.. [2] Lockwood, T. E.; de Vega, R. G.; Clases, D. An Interactive Python-Based Data Processing Platform for Single Particle and Single Cell ICP-MS. Journal of Analytical Atomic Spectrometry 2021, 36 (11), 2536–2544. https://doi.org/10.1039/D1JA00297J.
