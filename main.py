import re
from fractions import Fraction
from pathlib import Path
from pprint import pprint

import bs4
from bs4 import BeautifulSoup

DIR = Path.home() / "Downloads/com.flyersoft.moonreaderp"

PROG_FILE = DIR / "24.tag"

PROGRESS_RE = re.compile(
    r"""
    (^(?P<no1>[\d]+))           # unknown field 1
    (\@(?P<no2>[\d]+))?         # unknown field 2
    (\#(?P<no3>[\d]+))?         # unknown field 3
    (:(?P<percentage>[\d.]+))%  # ratio of already read pages
    """,
    re.VERBOSE,
)


def get_progress(progress_file: Path) -> dict[str, Fraction]:
    """For each book, get reading progress as a percentage."""
    progress = {}
    no_match = []

    progress_data = progress_file.read_bytes()

    soup = BeautifulSoup(progress_data, "xml")
    assert soup.map is not None, f"invalid progress_data {soup}"
    for entry in soup.map.find_all("string"):
        if not isinstance(entry, bs4.Tag):
            continue
        if not entry.string:
            continue
        file = entry["name"]
        if match := PROGRESS_RE.match(entry.string):
            val = Fraction(match["percentage"])
            if old := progress.get(file):
                # just to be extra sure
                print(f"duplicate! {file=}, {old=}, {val=}")
            progress[file] = val
        else:
            no_match.append(file)

    return progress


progress = get_progress(PROG_FILE)
pprint(progress)
