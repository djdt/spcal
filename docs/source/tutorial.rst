Tutorial: Basic usage of SPCal
==============================

Data import
-----------

The first step to performing any data processing is to load the data into SPCal.
SPCal supports import of delimited text files, which can be exported from instrument vendor software, and the raw data of Nu Instruments and TOFWERKs ICP-ToFs.
Data files can be loaded from the **File -> Open Sample File** or by drag-and-drop of files into the **Sample** tab


Text files
----------

.. figure:: images/tutorial_data_text_importer.png
   :width: 640px
   :align: center

   The import wizard for delimited text files.

Opening a text file will start the text import wizard. This wizard allows you to select which columns in the file to import and skip any non-data rows.
The wizard will attempt to guess the correct options when the file is loaded, A description of each option is shown in :numref:`tabtextoptions`.

.. _tabtextoptions:
.. list-table:: Options for the text import wizard.
    :header-rows: 1

    * - Option
      - Description
    * - Dwelltime
      - The instrument event acquistion time. Caculated from data when a column of with ``time`` in the title is included.
    * - Intensity Units
      - Selects if data is in *counts* or *counts-per-second (CPS)*. Defaults to counts unless *CPS* is included in the file header.
    * - Delimiter
      - The delimiter character.
    * - Import From Row
      - Skips the first *x* rows of the file (the header). The first non-skipped row should be the column names.
    * - Ingnore Columns
      - A ``;`` delimited list of which columns to ignore. For example time or index rows should not be imported.
