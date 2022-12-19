import argparse
import re
import sys
from pathlib import Path
from typing import List, Set


def collect_icons(path: Path) -> Set[str]:
    regex_icon = "(?:fromTheme|create_action)\\(\\s*['\"]([a-z\\-]+)['\"]"
    icons = set()

    for path in sorted(path.glob("**/*.py")):
        with path.open() as fp:
            icons.update(re.findall(regex_icon, fp.read()))
    return icons


def write_index_theme(path: Path, sizes: List[int], add_scaled: bool = True):
    directories = [str(size) for size in sizes]
    if add_scaled:
        directories.extend([dir + "@2" for dir in directories])
    with path.open("w") as fp:
        fp.write("[Icon Theme]\nName=spcal\nComment=Icons from KDE theme Breeze\n")
        fp.write(f"Directories={','.join(directories)}\n")
        fp.write("\n")
        for dir in directories:
            fp.write(
                f"[{dir}]\nSize={dir[:2]}\nScale={2 if '@' in dir else 1}\nType=Fixed\n\n"
            )


def write_qrc(
    qrc: Path,
    index: Path,
    icons: Path,
    icon_names: List[str],
    sizes: List[int],
    add_scaled: bool = True,
):
    with qrc.open("w") as fp:
        fp.write('<RCC version="1.0">\n')
        fp.write('<qresource prefix="icons/spcal/">\n')

        fp.write(f'\t<file alias="index.theme">{index}</file>\n')
        for icon in sorted(icon_names):
            match_found = False
            for size in sizes:
                matches = sorted(icons.glob(f"**/{size}/{icon}.svg"))
                if len(matches) > 0:
                    fp.write(f'\t<file alias="{size}/{icon}.svg">{matches[0]}</file>\n')
                    if add_scaled:
                        fp.write(f'\t<file alias="{size}@2/{icon}.svg">{matches[0]}</file>\n')
                    match_found = True
                    break
            if not match_found:
                print(f"warning: no match found for '{icon}'")
        fp.write("</qresource>\n")
        fp.write("</RCC>")


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
        default=[48, 32, 24],
        help="icon sizes to use, in order of preference",
    )
    parser.add_argument(
        "--add-scaled",
        action="store_true",
        help="add x 2 scaled icons",
    )
    args = parser.parse_args(sys.argv[1:])

    icon_names = collect_icons(args.project)
    write_index_theme(Path("index.theme"), args.sizes, args.add_scaled)
    write_qrc(Path("icons.qrc"), Path("index.theme"), args.icons, list(icon_names), args.sizes, args.add_scaled)
