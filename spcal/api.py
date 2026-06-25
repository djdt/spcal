import urllib.request
import json

SPCAL_RELEASE_URL = "https://api.github.com/repos/djdt/spcal/releases/latest"


def compare_version_strings(version_a: str, version_b: str) -> bool:
    """Returns true if version b is newer than version a

    Args:
        version_a: version string, X.Y.Z...
        version_b: version string, X.Y.Z...

    Returns:
        True if version_b > version_a
    """

    for a, b in zip(version_a.split("."), version_b.split(".")):
        ia, ib = int(a), int(b)
        if ib > ia:
            return True
        elif ia > ib:
            return False

    # in case a is 1.0, b is 1.0.1
    return version_b.count(".") > version_a.count(".")


def get_version_of_latest_release() -> str:
    """Retrieves the latest version of SPCal from the github.

    Returns:
        SPCal version in format "XX.XX.XX", "major.minor.release"
    Raises:
        TimeoutError: no connection after 5 seconds
        ConnectionError: status code is not 200, invalid connection
        ValueError: tag version  format is invalid
    """
    result = urllib.request.urlopen(SPCAL_RELEASE_URL, timeout=5)
    if result.status != 200:
        raise ConnectionError(f"invalid status code '{result.status}'")

    data = json.loads(result.read())
    version = data["tag_name"]
    if not version.startswith("v"):
        raise ValueError(f"version format is not valid '{version}'")
    return version[1:]
