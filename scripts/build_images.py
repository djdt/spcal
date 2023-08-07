import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List

import sympy

currie = "$$S_C = z_{1-\\alpha} \\sqrt{\\eta b + \\epsilon}$$"

formula_a = (
    "$$S_C = z_{1-\\alpha}"
    "\\sqrt{N_B \\frac{t_S}{t_B} \\left(1 + \\frac{t_S}{t_B}\\right)}$$"
)

formula_c = (
    "$$S_C = \\frac{z^{2}_{1-\\alpha}t_S}{2t_B}"
    "\\sqrt{\\frac{z^{2}_{1-\\alpha}t^{2}_{S}}{4t^{2}_{B}}"
    "+ N_B \\frac{t_S}{t_B} \\left(1 + \\frac{t_S}{t_B}\\right)}$$"
)

stapleton = (
    "$$S_C = \\frac{z_{1-\\alpha}}{4.112} \\left(\\frac{t_S}{t_B} - 1\\right)"
    "+ \\frac{z^{2}_{1-\\alpha}}{4}\\left(1+\\frac{t_S}{t_B}\\right)"
    "+ z_{1-\\alpha}\\sqrt{\\left(N_B+\\frac{z_{1-\\alpha}}{4.112}\\right)"
    "\\frac{t_S}{t_B}\\left(1+\\frac{t_S}{t_B}\\right)}$$"
)


def create_formula_images(path: Path) -> List[Path]:
    paths = []
    for formula, name in zip(
        [currie, formula_a, formula_c, stapleton],
        ["currie2008.png", "formula_a.png", "formula_c.png", "stapleton.png"],
    ):
        paths.append(path.joinpath(name))
        sympy.preview(
            formula,
            viewer="file",
            filename=path.joinpath(name),
            euler=True,
            dvioptions=["-D", "150"],
        )
    return paths


def write_qrc(qrc: Path, images: List[Path]):
    with qrc.open("w") as fp:
        fp.write('<RCC version="1.0">\n')
        fp.write('<qresource prefix="img/">\n')
        for image in images:
            fp.write(f'\t<file alias="{image.name}">{image}</file>\n')
        fp.write("</qresource>\n")
        fp.write("</RCC>")


def build_image_resource(qrc: Path, output: Path, rcc: str):
    cmd = [rcc, "-g", "python", "-o", str(output), str(qrc)]
    print(f"running {' '.join(cmd[:-1])} <{qrc.name}>")
    proc = subprocess.run(cmd, capture_output=True)
    proc.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path, help="The project directory.")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="additional images to add",
    )
    parser.add_argument(
        "--rcc", default="/usr/lib/qt6/rcc", help="rcc to generates resources"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="../spcal/resources/images.py",
        help="output path",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir, tempfile.NamedTemporaryFile() as qrc_tmp:
        images = create_formula_images(Path(tmp_dir))
        images.append(Path(__file__).parent.parent.joinpath("app.ico"))

        write_qrc(Path(qrc_tmp.name), images)
        build_image_resource(Path(qrc_tmp.name), args.output, args.rcc)
