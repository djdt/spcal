Viewing Data
============

.. _sample label:
.. figure:: images/tutorial_sample_tab.png
   :align: center

   The **Sample Tab** shows the currently loaded sample data. |c1| Current file, |c2| Current element, |c3| Sample parameters for selected element, |c4| Outputs for selected elements, |c5| Multi-element plot, |c6| Single element plot, |c7| Trim control, |c8| Legend, |c9| Reset zoom.


Files loaded into SPCal will be shown in the **Sample Tab**, as in :numref:`sample label`.
This tab shows the loaded signal data for all elements, and element specific options and outputs.
The currently selected element / name, as shown in |c2| :numref:`sample label`, can be edited to rename it.

The plot of sample data can be navigated using the left-mouse to scroll the view and mouse-wheel to zoom.
The y-axis of the plot is automatically scaled to show the maximum currently visible signal value.
The *trim controls* (|c7| :numref:`sample label`) can be used to trim data, limiting the region of analysis.
When the single element histogram is used (|c6| :numref:`sample label`) the mean and detection threshold are also shown.

Sample parameters
-----------------

:ref:`Calibration` requires the input sample of parameters for each element collected.
The parameters for the selected element (|c2| :numref:`sample label`) are shown on the left hand side of the tab (|c3| :numref:`sample label`) and summarised in :numref:`table sample`

.. _table sample:
.. list-table:: Sample parameters on the **Sample Tab**.
   :header-rows: 1

   * - Parameter
     - Description
   * - Density
     - The particle :term:`density`.
   * - Molar mass
     - The :term:`molar mass` (molecular weight) of the particle material.
   * - Ionic response
     - The detectors response for a given concentration of material.
   * - Mass fraction
     - The fraction of a particle mass represented by the measured element.

Buttons on the :term:`density`, :term:`molar mass`, :term:`ionic response` and :term:`mass fraction` will start the :ref:`Density Database`, :ref:`Mass Fraction Calculator` and :ref:`Ionic Response Calculator`.
When completed, these tools will automatically fill the corresponding field.
The :term:`ionic response` fields of the **Sample Tab** and **Reference Tab** are linked, and editing one value will also change the other.


.. |c1| unicode:: U+2460
.. |c2| unicode:: U+2461
.. |c3| unicode:: U+2462
.. |c4| unicode:: U+2463
.. |c5| unicode:: U+2464
.. |c6| unicode:: U+2465
.. |c7| unicode:: U+2466
.. |c8| unicode:: U+2467
.. |c9| unicode:: U+2468
