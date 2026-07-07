import urllib.request
import json

API_URL = "https://api.github.com/repos/djdt/spcal/"


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


def get_github_release_info(name: str = "latest") -> dict:
    """Retrieves the release info from the SPCal GitHub.

    The JSON format can be found in the `API docs <https://docs.github.com/en/rest/releases/releases>`_.

    Args:
        name: either 'latest' or a tag in the format 'tags/vXX.XX.XX'.

    Returns:
        JSON style dictionary of release information

    Raises:
        TimeoutError: no connection after 5 seconds
        ConnectionError: status code is not 200, invalid connection
        ValueError: tag version  format is invalid
    """
    result = urllib.request.urlopen(API_URL + "releases/" + name, timeout=5)
    if result.status != 200:  # pragma: no cover, error
        raise ConnectionError(f"invalid status code '{result.status}'")

    return json.loads(result.read())
