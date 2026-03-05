Data Export
===========

.. _export options:
.. figure:: ../images/usage_export.png
   :align: center

   The export options dialog allows you to select what data is exported.

Results of the current file can be exported via **File -> Export Results**.
This opens the dialog shown in :numref:`export options`, where options for exporting can be entered.
The filename of the export is set to *Save File*, and can be selected by pressing *Select File*.
By default the export name will be the imported sample filename with ``_spcal_results.csv`` appended.
The units of exported masses, sizes and concentrations can be set using the *Units* controls, and default to fg and nm.
Finally, the *Export options* control what extra data is written to the file.
All exports will contain particle numbers, concentrations, backgrounds, the mean and median of signal, size and mass, and limits of detection.

Checking *Instrument, limit and isotope options* will save parameters such as the :term:`event time`, instrument :term:`uptake` and particle :term:`density` to the file.
Checking *Paraticle data arrays* will save the particle times, signals and any calibrated data to the end of the file.
Each row of this data is a single detection with columns of elements and units, particles may contain one or more elements and are blank when that element is not detected. An example is shown below.

.. code-block:: 

    # Raw detection data
    Time (s),27Al (cts),28Si (cts),56Fe (cts),56Fe (kg)
    1.162329,1477.1984 ,3324.832  ,285.52759 ,8.389378E-19
    1.229459,31.392302 ,          ,          ,
    1.368913,          ,251.97623 ,820.53406 ,2.410895E-18
    1.371975,          ,491.5755  ,          ,
    1.606808,          ,          ,283.00967 ,8.3153966E-19
    1.714657,          ,          ,66.157867 ,1.9438519E-19


The last export option, *Particle compositions*, will add the particle composition clusters to the file, shown below.
Each row contains the mean fraction (and error) for each element and the number of particles (``count``) per cluster.
This option will also add the cluster index (which cluster a particle belongs to) as a column to the data array export.

.. code-block::

    # Compositions (signal),Count,Al27 mean (cts), Al27 std (cts),...
    # 0                    ,20241,0.1261         ,0.1584         ,...
    # 1                    ,1682 ,1              ,0              ,...
    # 2                    ,5    ,0.7344         ,0.004635       ,...
    # 3                    ,3    ,0.7838         ,0.003697       ,...
