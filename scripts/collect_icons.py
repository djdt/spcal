import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Set


def collect_icons(path: Path) -> Set[str]:
    regex_icon = "(?:fromTheme|create_action)\\(\\s*['\"]([a-z\\-]+)['\"]"
    icons = set()

    for path in sorted(path.glob("**/*.py")):
        with path.open() as fp:
            icons.update(re.findall(regex_icon, fp.read()))
    return icons


def write_index_theme(path: Path, sizes: list[int]):
    directories = [f"{size}x{size}" for size in sizes]
    directories.extend([dir + "@2" for dir in directories])
    with path.open("w") as fp:
        fp.write("[Icon Theme]\nName=spcal\nComment=Icons from KDE theme Breeze\n")
        fp.write(f"Directories={','.join(directories)}\n")
        fp.write("\n")
        for size in sizes:
            fp.write(f"[{size}x{size}]\nSize={size}\nType=Fixed\n\n")
            fp.write(f"[{size}x{size}@2]\nSize={size}\nScale=2\nType=Fixed\n\n")


def write_qrc(
    qrc: Path,
    index: Path,
    icons: Path,
    icon_names: list[str],
    sizes: list[int],
):
    with qrc.open("w") as fp:
        fp.write('<RCC version="1.0">\n')
        fp.write('<qresource prefix="icons/spcal/">\n')

        fp.write(f'\t<file alias="index.theme">{index}</file>\n')
        for icon in sorted(icon_names):
            match_found = False
            for size in sizes:
                for match in sorted(icons.glob(f"**/{size}/{icon}.svg")):
                    fp.write(
                        f'\t<file alias="{size}x{size}/{icon}.svg">{match}</file>\n'
                    )
                    fp.write(
                        f'\t<file alias="{size}x{size}@2/{icon}.svg">{match}</file>\n'
                    )
                    match_found = True
                    break
            if not match_found:
                print(f"warning: no match found for '{icon}'")
        fp.write("</qresource>\n")
        fp.write("</RCC>")


def build_icons_resource(qrc: Path, output: Path, rcc: str):
    cmd = [rcc, "-g", "python", "-o", str(output), str(qrc)]
    print(f"running {' '.join(cmd[:-1])} <icons.qrc>")
    proc = subprocess.run(cmd, capture_output=True)
    proc.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path, help="The project directory.")
    parser.add_argument(
        "--icons",
        type=Path,
        default="/usr/share/icons/breeze",
        help="the icons directory",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[16, 24, 32],
        help="icon sizes to use",
    )
    parser.add_argument(
        "--rcc", default="/usr/lib/qt6/rcc", help="rcc to generates resources"
    )
    parser.add_argument(
        "--output", type=Path, default="../spcal/resources/icons.py", help="output path"
    )
    args = parser.parse_args(sys.argv[1:])

    icon_names = collect_icons(args.project)
    print(f"found {len(icon_names)} icons", flush=True)
    with tempfile.NamedTemporaryFile() as index_tmp, tempfile.NamedTemporaryFile() as qrc_tmp:
        index, qrc = Path(index_tmp.name), Path(qrc_tmp.name)
        write_index_theme(index, args.sizes)
        write_qrc(qrc, index, args.icons, list(icon_names), args.sizes)
        build_icons_resource(qrc, args.output, args.rcc)
