import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta as td
from fractions import Fraction
from pathlib import Path
from pprint import pprint

import bs4
from bs4 import BeautifulSoup

DIR = Path.home() / "Downloads/com.flyersoft.moonreaderp"

PROG_FILE = DIR / "24.tag"
DB_PATH = DIR / "39.tag"

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


DAILY_STATS_RE = re.compile(r"(?P<day_delta>\d+)\|(?P<time_ms>\d+)@(?P<words>\d+)")
UNIX_EPOCH = date.fromtimestamp(0)
TODAY = date.today()


@dataclass(slots=True, frozen=True)
class DailyStats:
    day: date
    reading_time: td
    num_words: int

    @staticmethod
    def from_str(val: str) -> "DailyStats":
        match = DAILY_STATS_RE.match(val)
        if match is None:
            raise ValueError(f"invalid daily stats string {val}")
        day = UNIX_EPOCH + td(days=int(match["day_delta"]))
        assert day <= TODAY, f"Got future day {day} in daily stats"
        reading_time = td(milliseconds=int(match["time_ms"]))
        words = int(match["words"])
        return DailyStats(day, reading_time, words)


@dataclass(slots=True, frozen=True)
class BookReadStats:
    reading_time: td
    read_words: int
    daily_stats: list[DailyStats]


def get_daily_progress(db_con: sqlite3.Connection):
    cur = db_con.execute("SELECT * FROM 'statistics'")
    stats = {}
    for row in cur.fetchall():
        file = row['filename']
        reading_time = td(milliseconds=row['usedTime'])
        all_stats = [DailyStats.from_str(line) for line in row['dates'].splitlines()]
        book_stats = BookReadStats(reading_time, row['readWords'], all_stats)
        if old := stats.get(file):
            # just to be extra sure
            print(f"duplicate! {file=}, {old=}, {book_stats=}")
        stats[file] = book_stats
    cur.close()
    return stats


def main():
    progress = get_progress(PROG_FILE)
    pprint(progress)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    daily_progress = get_daily_progress(con)
    pprint(daily_progress)


if __name__ == "__main__":
    main()
