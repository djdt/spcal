from pathlib import Path

import nanopart
from nanopart.io import read_nanoparticle_file

from nanopart.calc import calculate_limits


# def export_results(path: Path,
#             f"Detected particles,{self.sizes.size}\n"
#             f"Number concentration,{self.number.value()},{self.number.unit()}\n"
#             f"Concentration,{self.conc.value()},{self.conc.unit()}\n"
#             f"Ionic background,{self.background.value()},{self.background.unit()}\n"
#             f"Mean NP size,{np.mean(self.sizes) * 1e9},nm\n"
#             f"Median NP size,{np.median(self.sizes) * 1e9},nm\n"
#             f"LOD equivalent size,{self.background_lod_size * 1e9},nm\n"
#         )

#         header = text + "Masses (kg),Sizes (m)"
#         data = np.stack((self.masses, self.sizes), axis=1)

#         np.savetxt(
#             path,
#             data,
#             delimiter=",",
#             header=header,
#         )
# )


def process_file_mass_response(
    file: Path,
    dwelltime: float,
    density: float,
    molarratio: float,
    massresponse: float,
    options: dict,
) -> bool:
    responses, _ = read_nanoparticle_file(file, delimiter=",")

    if responses is None or responses.size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(
        responses, options["limit_method"], options["sigma"], options["epsilon"]
    )

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")


def updateResultsMassResponse(self, massresponse: float) -> None:
    density = self.sample.density.baseValue()
    molarratio = float(self.sample.molarratio.text())

    self.masses = self.sample.detections * (massresponse / molarratio)
    self.sizes = nanopart.particle_size(self.masses, density=density)

    self.number_concentration = None
    self.concentration = None
    self.ionic_background = None

    self.background_lod_mass = self.sample.limits[3] / (massresponse * molarratio)
    self.background_lod_size = nanopart.particle_size(
        self.background_lod_mass, density=density
    )


def updateResultsNebEff(self, efficiency: float) -> None:
    dwelltime = self.options.dwelltime.baseValue()
    density = self.sample.density.baseValue()
    molarratio = float(self.sample.molarratio.text())
    time = self.sample.timeAsSeconds()
    uptake = self.options.uptake.baseValue()
    response = self.options.response.baseValue()

    self.masses = nanopart.particle_mass(
        self.sample.detections,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        molar_ratio=molarratio,
    )
    self.sizes = nanopart.particle_size(self.masses, density=density)

    self.number_concentration = nanopart.particle_number_concentration(
        self.sample.detections.size,
        efficiency=efficiency,
        flowrate=uptake,
        time=time,
    )
    self.concentration = nanopart.particle_total_concentration(
        self.masses,
        efficiency=efficiency,
        flowrate=uptake,
        time=time,
    )

    self.ionic_background = self.sample.background / response
    self.background_lod_mass = nanopart.particle_mass(
        self.sample.limits[3],
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        molar_ratio=molarratio,
    )
    self.background_lod_size = nanopart.particle_size(
        self.background_lod_mass, density=density
    )
