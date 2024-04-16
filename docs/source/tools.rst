Tools
=====


Ionic Response Dialog
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

   The **Ionic Response Dialog**. 

The **Ionic Response Dialog** is used to calculate the :term:`ioinic response` using one or more data files.
