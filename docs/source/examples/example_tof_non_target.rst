Non-targetted Analysis Using an ICP-ToF
=======================================


The simultaneous acquistion of the full elemental mass range makes ICP-ToF an excellent particle screening tool.
In this example, elements of interest are automatically selected using parameters set during import.


#. Download the required data file.
    The ``tof_non_target.csv`` file can be found in the `example_3_data.zip <https://github.com/djdt/spcal/docs/data/example_3_data.zip>`_ archive.

#. Import the sample file.
    Open the sample file to start the :red:`Data Import` wizard.
    Select *Non-target screening* and enter 100 into the *Screening ppm* field, this will select elements with greater than 100 particles per million events.
    Press *Ok* to start the screening process.

#. Verify element selection.
    The non-target screening will automatically select all elements with detection counts above the given *Screening ppm* value.
    These element are also colour coded (using the *viridis* colour scale) to give an idea on the relative number of particles.
