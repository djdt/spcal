Calibrating Quadrupole ICP-MS Data
==================================

This example will guide you through importing, calibrating and exporting single-particle data collected on a quadrupole ICP-MS.
A 15 nm gold NP is used as a sample, with a 50 nm gold NP as a reference.
A sample is loaded and then calibrated using the *Reference Particle* method described in :ref:`Calibration`.

.. _table parameters:
.. list-table:: Parameters used in this example.
    :header-rows: 1

    * - Parameter
      - Location
      - Value
    * - Uptake
      - Options
      - 0.35 ml/min
    * - Density
      - Sample 
      - 19.3 g/cm3 (Au)
    * - Ionic response
      - Sample
      - 17.5 counts/ug
    * - Density
      - Reference 
      - 19.3 g/cm3 (Au)
    * - Ionic response
      - Reference
      - 17.5 counts/ug
    * - Diameter
      - Reference
      - 50 nm


#. Download the required sample and reference data files.
    The ``quad_sample_15nm.csv`` and ``quad_reference_50nm.csv`` files are available as a Zip archive `example_1_data.zip <https://github.com/djdt/djdt.github.io/blob/main/spcal_example_data/example_1_data.zip>`_ on the GitHub. 

#. Enter the instrument parameters.
    In the **Options Tab**, enter the :term:`uptake` from :numref:`table parameters`.

#. Import the sample file.
    The ``quad_sample_15nm.csv`` file can be opened using **File -> Open Sample File**, then switch to the **Sample Tab**.
    Enter the sample parameters listed in :numref:`table parameters`.
    A detailed guide on the :ref:`Data Import` wizard is available.

    .. _quad sample tab:
    .. figure:: ../images/example_quad_sample_tab.png
       :width: 60%
       :align: center

       The sample tab after importing ``quad_sample_15nm.csv`` and entering the sample parameters.

    Your sample tab should look like :numref:`quad sample tab`.

#. Set the :term:`transport efficiency` method.
    In the **Options Tab** to *Reference Particle*.
    This enables the **Reference Tab**.

#. Import the reference file.
    Open ``quad_reference_50nm.csv`` using **File -> Open Reference File**, and switch to the **Reference Tab**.
    Enter the reference parameters listed in :numref:`table parameters`.
    A detailed guide on the :ref:`Data Import` wizard is available.

    .. _quad reference tab:
    .. figure:: ../images/example_quad_reference_tab.png
       :width: 60%
       :align: center

       The reference tab after importing ``quad_reference_50nm.csv`` and entering the reference parameters.

    Your reference tab should look like :numref:`quad reference tab` and the :term:`transport efficiency` calculated as ~0.01897.

#. Switch to the **Results Tab**.
    .. _quad results tab:
    .. figure:: ../images/example_quad_results_tab.png
       :width: 60%
       :align: center

       The results tab showing the sample size distribution, available after calibrating.

    The calculated results for the loaded sample are shown as in :numref:`quad results tab`.
    Switch to *Size* mode, the median size should be around 15 nm.

#. Export the results.
    Press the *Export Results* button to save the results to a file.
