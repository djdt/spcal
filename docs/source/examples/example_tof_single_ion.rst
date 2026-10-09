Per-mass Single Ion Area on an ICP-ToF
======================================

Determining the :term:`single ion area` (SIA) is essential for accurate thresholding of ICP-ToF data.
In SPCal we approximate the SIA using a shape parameter, :math:`\sigma`.
This value can be determined from ionic data using the methods described in :ref:`Recovery of compound-Poisson-lognormal parameters`.
In the SPCal GUI we can determine SIA values for each mass by loading a low concentration ionic standard (1 - 10 ppb) into the :ref:`Single Ion Distribution Dialog`, found in the **Limit Options Dock**.


#. Download the required data file.
    The ``tof_single_ion`` directory can be found in the `example_4_data.zip <https://github.com/djdt/djdt.github.io/raw/main/spcal_example_data/example_4_data.zip>`_ archive.

#. Start the :ref:`Single Ion Distribution Dialog`
   Click the *Single Ion Options...* button in the **Compound** tab of the **Limit Options Dock**.
   This will start the dialog.

#. Open the example data file.
   Once started, the dialog will prompt you to load a file.
   Load the ``tof_single_ion/run.info`` file.

    .. _tutorial single ion:
    .. figure:: ../images/usgae_single_ion.png
       :width: 60%
       :align: center

       The single ion dialog.

   The dialog should now look like :numref:`tutorial single ion`, a scatter plot of masses and calcualated shapes.
   Two red lines show the IQR of the expected shape values, and are calculated from data collected on several Vitesse instruments.

#. Check the calculated shape values for anomalies.
   The shapes are shown as a scatter plot on the right half of the dialog, as in :numref:`tutorial single ion`.
   Yellow points have been excluded due to low or high zero counts (preventing calculation of :math:`\lambda`), by default a maximum error of 1% is allowed.
   Red points are excluded due to presence of particles, for exmaple both the silver isotopes (107 and 109).
   *Middle-click* on either point to display the signals and confirm that particles are causing high variance and incorrect retreival of the SIA shape.

#. *Left-click* any point to toggle its selection.
   Masses that are not-selected will default to the :math:`\sigma` value provided in the **Compound** tab of the **Limit Options Dock**.
   You can also select isotopes using the *Select isotopes* button.
   By default all valid masses (correct zero counts and no particles) that correspond to an isotope with greater than 10 % natural abundance are selected.

#. Apply the dialog.
   Clicking *Apply* will use the per-mass SIA for selected isotopes / masses.
