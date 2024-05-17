:orphan:

Glossary
========

.. glossary::

    accumulation threshold
        The value above which contiguous regions with at least one detection are summed.

    diameter
        Reference particle diameter.
        Must be determined externally.

    density
        Particle density.
        Must be measured externally or assumed for a certain material.
        The :ref:`Density Database` contains the density for several hundred materials.

    detection threshold
        The value above which a signal is considered a detection (particle).
        This is also called the *critical value* in Poisson statistics.

    dwelltime
        The total acquisition time for a single event.
        The dwelltime is set during data import and cannot be edited.

    error rate
        Determines the number of false detections, i.e. background signal incorrectly identified as particles.
        An error rate (:math:`\alpha`) of :math:`10^(-6)` corresponds to 1 false detection per 1 million events.

    ionic response
        The signal produced per mass of material.
        Measured by collecting one or more ionic standards of known concentrations.
        The ionic response can be calibrated from multiple standard levels using the :ref:`Ionic Response Calculator`.

    mass fraction
        The amount of measured material per total particle.
        If an AuAg particle is 40 % Au by mass, then the mass fraction for Au will be 0.4 and 0.6 for Ag.
        The :ref:`Mass Fraction Calculator` can calculate the mass fraction from a known molecular formula.

    mass response
        An alternate calibration strategy.
        The mass response uses the average mass of a reference particle to calibrate signal into mass.

    molar mass
        The molecular weight of the particle material.
        Used to calculate the intracellular concentration for single-cell ICP-MS.

    transport efficiency
        The fraction of material that is successfully transported to the detector.
        Typically 0.02 - 0.1 (2 - 10%).

    uptake
        The sample flow rate.
        This can be measured gravimetrically, via the change in sample mass over a fixed time.
