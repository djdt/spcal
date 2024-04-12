Compound Poisson Thresholds
===========================

To achieve sufficient resolution a time-of-flight instrument cannot be operated in pulse-counting mode, where the electron pulses from indiviual ions are counted as discrete events.
Instead, the detectors in TOFs use the raw output of fast analouge-to-digital converters. This exposes the variation in current produced by an electron multiplier for a single ion, known as the pulse-height distribution (PHD) or single-ion current. The result of multiple ions striking the detector is therefore a Poisson sampling of the PHD. During calibration of the detector the single-ion area (SIA) must be determined. Typically this is performed by analysing a very low concentration sample, one that is likely to only produce single-ion events. The raw data from the detector is then normalised to approxmate the signal produced for 1 ion (count) by dividing by the mean of the recorded SIA.


.. centered::
   |integer_data|

The Poisson thresholding typically used for spICP-MS is not valid for spICP-ToF, as is easily established by looking at spICP-ToF data. Unlike data from a quadrupole instrument, this data is non-integer. Instead compound-Poisson sampling must be used, ideally of the actual SIA of the instrument.

.. |integer_data| image:: images/integer_data.png
    :width: 640px
