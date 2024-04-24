Particle Compositions on an ICP-ToF
===================================

The simultaneous aquistion of elements by an ICP-ToF has major benefits over a quadrupole based instrument.
One key advantage is the ability to determine particle compositions.
In this example a mixture of gold, silver and gold-silver core-shell particles are analysed, and separated using clustering.


#. Download the required data file.
    The ``tof_mix_au_ag_auag.csv`` file can be found in the `example_2_data.zip <https://github.com/djdt/spcal/docs/data/example_2_data.zip>`_ archive.

#. Import the data file.
    In this example we are only interested in the gold and silver isotopes (Ag107, Ag109 and Au197).
    Using the :ref:`Data Import` wizard, filter out the unused isoptes by entering their columns (``1;2;3;4;5;6;7;8``) into the *Ignore Columns* field.

    .. _tof2 sample tab:
    .. figure:: ../images/example_tof_cluster_sample_tab.png
       :width: 60%
       :align: center

       The sample tab after importing the gold and silver isotope data.

#. Switch to the **Results Tab** and select the :ref:`Compositions` view.
    A detailed description of the results tab can be found in the :ref:`Processing Results` section.

    .. _tof2 results pre:
    .. figure:: ../images/example_tof_cluster_results_1.png
       :align: center

#. Lower the minimum cluster size.
    The default parameters limit the minmum cluster size to 5% of the total particle count.
    In this example the number of AuAg core-shell aprticles is small so we need to decrease it.
    In the graph options dialog, set the *Minimum cluster size* to 100.

    .. _tof2 results post:
    .. figure:: ../images/example_tof_cluster_results_2.png
       :align: center

       Three particle compositions are visible, gold, silver and gold-silver core-shell.

#. Filter results to include only pure gold particles.
    We can use the clustering results to limit our analysis to a single particle type.
    Open the **Filter Dialog** and add a cluter filter for cluster index 2, see :ref:`Filtering` for details.
    The cluster index for each cluster can be found below the pie in the Composition View, as in :numref:`tof2 results post`.

#. Switch to the Histogram view.
    The displayed histogram for gold is now free of interfering signals from core-shell particles.

    .. _tof2 filtered:
    .. figure:: ../images/example_tof_cluster_filter.png
       :align: center

       Filtering (right) has removed the gold signals from core-shell particles, leaving only signals from pure gold particles.
