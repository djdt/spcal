from PySide2 import QtCore, QtWidgets

from futures.conn
import numpy as np
from pathlib import Path

import nanopart

from nanopart.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from nanopart.io import read_nanoparticle_file, export_nanoparticle_results

from nanopart.gui.inputs import SampleWidget, ReferenceWidget
from nanopart.gui.options import OptionsWidget

from typing import List


class BatchProcessDialog(QtWidgets.QDialog):
    def __init__(
        self,
        reference: ReferenceWidget,
        options: OptionsWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init(parent)
        self.reference = reference
        self.options = options

    def dialogLoadFiles(self) -> None:
        files, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Batch Process Files", "", "CSV Documents (.csv)"
        )
        if len(files) > 0:
            self.batchProcess([Path(file) for file in files])

    # def batchProcess(self, files: List[Path]) -> None:
    #     with ProcessPoolExecutor() as executor:
    #         pass
