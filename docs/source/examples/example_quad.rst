Calibrating Quadrupole ICP-MS Data
==================================

This example will guide you through importing, calibrating and exporting single-particle data collected on a quadrupole ICP-MS.
A 15 nm gold NP is used as a sample, with a 50 nm gold NP as a reference.
A sample is loaded and then calibrated using the :ref:`Reference Particle` method described in :ref:`Calibration`.

.. _table parameters:
.. list-table:: Parameters used in this example.
    :header-rows: 1

    * - Parameter
      - Location
      - Value
    * - Uptake
      - Instrument Options
      - 0.35 ml/min
    * - Density
      - Isotope Options 
      - 19.3 g/cm3 (Au)
    * - Ionic response
      - Isotope Options
      - 17.5 counts/ug (Au)
    * - Mass Fraction
      - Isotope Options
      - 1.0 (Au)
    * - Diameter
      - Transport Efficiency Calculator
      - 50 nm


#. Download the required sample and reference data files.
    The ``quad_sample_15nm.csv`` and ``quad_reference_50nm.csv`` files are available as a Zip archive `example_1_data.zip <https://github.com/djdt/djdt.github.io/raw/main/spcal_example_data/example_1_data.zip>`_ on the GitHub. 
    Download and extract the files.

#. Import the sample file and reference files.
    The ``quad_sample_15nm.csv`` and ``quad_reference_50nm.csv`` files can be opened using **File -> Open Sample File**.
    Enter the sample parameters listed in :numref:`table parameters`.
    A detailed guide on the :ref:`Data Import` wizard is available.

    .. _tutorial quad options:
    .. figure:: ../images/tutorial_quad_options.png
       :width: 60%
       :align: center

       The sample tab after importing ``quad_sample_15nm.csv`` and entering the sample parameters.

    SPCal should look like :numref:`tutorial quad options`.

#. Enter the experiment parameters.
    In the **Instrument Options Dock**, enter the :term:`uptake` from :numref:`table parameters`.
    Enter the :term:`density`, :term:`ionic response` and :term:`mass fraction` into the **Isotope Options Dock**.
    The button next to *Trans. Efficiency* should now be active.

#. Calculate the :term:`transport efficiency`.
    *Ensure that the reference data file is currently selected* in the **Data Files Dock**!
    Press to button next to the *Trans. Eddiciency* field to open the **Transport Efficiency Calculator**.
    Most values should be pre-filled from the existing method.
    Enter the reference particle diameter listed in :numref:`table parameters`.
    The calculated efficiency should be close to 0.0185.

#. Check results and size histograms
   *Switch to the sample file* in the **Data Files Dock**.
   Set the current *Key* from *Signal* to *Size* using the controls in the top toolbar.
   The median and mean size listed in the **Results Dock** should be close to 15 nm.

    .. _tutorial quad results:
    .. figure:: ../images/tutorial_quad_results.png
       :width: 60%
       :align: center

       SPCal histogram view after importing ``quad_sample_15nm.csv`` and calibrating into sizes.

   The side toolbar can be used to show a histogram of the particle sizes, as in :numref:`tutorial quad results`.


#. Export the results.
    Press **File -> Export Results** button to save the results to a file.
