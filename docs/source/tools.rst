Tools
=====


Ionic Response Calculator
---------------------

.. _ionic response example:
.. figure:: images/tools_ionic_response_example.png
    :align: center

    Measurement of a dissolved standard, the mean of the signal (red line) can be used to determine the :term:`ionic response`.

To calculate the :term:`ionic response` one or more dissolved standards must be run under the same conditions as the spICP-MS sample.
When one standard is used, the :term:`ionic response` is taken as the mean value of the data, :numref:`ionic response example`.
When two or more are used, a calibration curve is created, with the :term:`ionic response` taken as the slope of that curve.

Ideally a blank and several dissolved standards are analysed, covering the range of signal produced by nanoparticles in the sample.
In practice, a blank and one standard are usually sufficient.
As the :term:`ionic response` will change day-to-day with instrument conditions so should be determined every run.

.. _ionic response dialog:
.. figure:: images/tools_ionic_response_dialog.png
   :align: center

   The **Ionic Response Calculator**. |c1| The currently loaded data and mean (red), |c2| Calibration curve, |c3| Concentration table, |c4| Import data for new level, |c5| Concetration units.

The **Ionic Response Calculator** is used to calculate the :term:`ioinic response` using one or more data files.
To use the calculator import one or more data files (|c4| :numref:`ionic response dialog`) and set their corresponding concentrations in the concentration table (|c3| :numref:`ionic response dialog`).
For information on importing data see :doc:`Data Import <data_import>`.

Once all levels are imported the :term:`ionic response` (the calibration slope) and calibrations can be exported using the *Save* button.
If the **Ionic Response Calculator** was started from an :term:`ionic response` field in the **Sample** or **Reference Tab** then pressing *Ok* will automatically fill that field.
