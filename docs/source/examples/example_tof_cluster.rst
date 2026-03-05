Particle Compositions on an ICP-ToF
===================================

The simultaneous aquistion of elements by an ICP-ToF has major benefits over a quadrupole based instrument.
One key advantage is the ability to determine particle compositions.
In this example a mixture of gold, silver and gold-silver core-shell particles are analysed, and separated using clustering and compositional filtering.


#. Download the required data file.
    The ``tof_mix_au_ag_auag.csv`` file can be found in the `example_2_data.zip <https://github.com/djdt/djdt.github.io/raw/main/spcal_example_data/example_2_data.zip>`_ archive.

#. Import the data file.
    In this example we are only interested in the gold and silver isotopes (Ag107, Ag109 and Au197).
    Using the :ref:`Data Import` wizard, filter out the unused isoptes (transition metals) by unchecking their columns.

#. Select the :ref:`Compositions` view.
    A detailed description of the view can be found in the :ref:`Processing Results` section.
    In :numref:`tof cluster results 1` we can see two clusters, representing pure gold and pure silever particles.
    The core-shell are fewer in number that the default cutoff (5% of the maximum cluster) so are hidden.

    .. _tof cluster results 1:
    .. figure:: ../images/tutorial_tof_cluster_results_1.png
       :align: center

       Composition clusters, only two are visible.

#. Lower the minimum cluster size.
    The default parameters limit the minmum cluster size to 5% of the total particle count.
    In this example the number of AuAg core-shell aprticles is small so we need to decrease it.
    In the graph options dialog, set the *Minimum cluster size* to 100.

    .. _tof cluster results 2:
    .. figure:: ../images/tutorial_tof_cluster_results_2.png
       :align: center

       Three particle compositions are visible, gold, silver and gold-silver core-shell.

#. Filter results to include only pure gold particles.
    We can use the clustering results to limit our analysis to a single particle type.
    Open the **Filter Dialog** and add a cluter filter for cluster index 2 (gold particles), see :ref:`Filtering` for details.
    The cluster index for each cluster can be found below the pie in the Composition View, as in :numref:`tof cluster results 2`.

    .. note::
        A similar result could be obtained by filtering for particles that contain no silver (Ag signal == 0).
        

#. Switch to the :ref:`Histograms` view.
    The displayed histogram for gold is now free of interfering signals from core-shell particles.

    .. _tof cluster filtered:
    .. figure:: ../images/tutorial_tof_cluster_filtered.png
       :align: center

       Filtering (right) has removed the gold signals from core-shell particles, leaving only signals from pure gold particles.
