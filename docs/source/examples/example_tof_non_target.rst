Non-targeted Analysis Using an ICP-ToF
======================================


The simultaneous acquisition of the full elemental mass range makes ICP-ToF an excellent particle screening tool.
In this example, elements of interest in a soil extract are automatically selected using parameters set during import.
A full description of the non-target screening tool is available at :ref:`Non-target Screening`.


#. Download the required data file.
    The ``tof_non_target.csv`` file can be found in the `example_3_data.zip <https://github.com/djdt/djdt.github.io/blob/main/spcal_example_data/example_3_data.zip>`_ archive.

#. Import the sample file.
    Open the sample file to start the :red:`Data Import` wizard.
    Select *Non-target screening* and enter 1000 into the *Screening ppm* field, this will select elements with greater than 1000 particles per million events.
    Press *Ok* to start the screening process.

#. Verify element selection.
    .. _nontarget_dialog:
    .. figure:: ../images/example_tof_non_target_dialog.png
       :width: 60%
       :align: center
       
       Elements selected during the screening are colour coded using the *viridis* colour scale to show the relative number of detected particles.

    The non-target screening will automatically select all elements with detection counts above the given *Screening ppm* value, in this case Fe, La and Ce.
    Press *Ok* to import the data.
