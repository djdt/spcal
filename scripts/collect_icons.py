import argparse
import re
import sys
from pathlib import Path
from typing import List, Set

# python collect_icons.py .. --exclude 64 96
# /usr/lib/qt6/rcc scripts/icons.qrc -g python -o spcal/resources/icons.py


def collect_icons(path: Path) -> Set[str]:
    regex_icon = "(?:fromTheme|create_action)\\(\\s*['\"]([a-z\\-]+)['\"]"
    icons = set()

    for path in sorted(path.glob("**/*.py")):
        with path.open() as fp:
            icons.update(re.findall(regex_icon, fp.read()))
    return icons


def write_qrc(
    qrc: Path, icons: Path, reroot: Path, exclude: List[Path], icon_names: List[str]
):
    with qrc.open("w") as fp:
        fp.write("<RCC>\n")
        fp.write("<qresource>\n")

        theme = list(icons.glob("**/index.theme"))[0]
        fp.write(f'\t<file alias="{theme.relative_to(reroot)}">{theme}</file>\n')

        for path in sorted(icons.glob("**/*.svg")):
            if any(parent in exclude for parent in path.parents):
                continue
            if path.stem in icon_names or path.name == "index.theme":
                fp.write(f'\t<file alias="{path.relative_to(reroot)}">{path}</file>\n')
        fp.write("</qresource>\n")
        fp.write("</RCC>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path, help="The project directory.")
    parser.add_argument(
        "--icons",
        type=Path,
        default="/usr/share/icons/breeze",
        help="The icons directory",
    )
    parser.add_argument(
        "--reroot",
        type=Path,
        default="/usr/share",
        help="Path to remove, relative to icon paths.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[],
        help="Exclude icons from any given paths.",
    )
    args = parser.parse_args(sys.argv[1:])

    icon_names = collect_icons(args.project)
    write_qrc(
        Path("icons.qrc"), args.icons, args.reroot, args.exclude, list(icon_names)
    )
