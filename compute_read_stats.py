#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beautifulsoup4 >= 4.11.1",
#     "jinja2 >= 3.1.2",
#     "gitignore-parser >= 0.1.3",
#     "lxml >= 5.3.0",
#     "rich>=14.2.0",
#     "tomli >= 2.0.1; python_version < '3.11'",
# ]
# ///

import argparse
import difflib
import json
import logging
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from datetime import timedelta as td
from fractions import Fraction
from functools import partial
from itertools import combinations, starmap
from os import PathLike
from pathlib import Path, PosixPath, PurePosixPath
from typing import Callable, Literal

import bs4
from bs4 import BeautifulSoup
from gitignore_parser import parse_gitignore
from rich.logging import RichHandler
from rich.pretty import pretty_repr

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=[RichHandler()])

PROGRESS_RE = re.compile(
    r"""
    (^(?P<no1>[\d]+))           # unknown field 1
    (\@(?P<no2>[\d]+))?         # unknown field 2
    (\#(?P<no3>[\d]+))?         # unknown field 3
    (:(?P<percentage>[\d.]+))%  # ratio of already read pages
    """,
    re.VERBOSE,
)

DAILY_STATS_RE = re.compile(r"(?P<day_delta>\d+)\|(?P<time_ms>\d+)@(?P<words>\d+)")
UNIX_EPOCH = date.fromtimestamp(0)
TODAY = date.today()

IGNOREFILE_NAME = "excluded.gitignore"
MR_ATTACH_DIR = "/sdcard/"
"""This will be considered as the base path when deciding which files to ignore"""
IgnoreMatcher = Callable[[PathLike], bool]
DEFAULT_IGNORE_MATCHER: IgnoreMatcher = lambda _: False


@dataclass(slots=True, frozen=True, unsafe_hash=True)
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
        reading_time = td(milliseconds=int(match['time_ms']))
        words = int(match['words'])
        return DailyStats(day, reading_time, words)

    def __str__(self) -> str:
        return f"read {self.num_words} words on {self.day.isoformat()} in {self.reading_time.total_seconds()} seconds"


@dataclass(slots=True, frozen=True)
class BookReadStats:
    reading_time: td
    read_words: int
    daily_stats: list[DailyStats]


@dataclass(slots=True, frozen=True, unsafe_hash=True)
class BookMetadata:
    title: str
    author: str


type OverallProgressDict = dict[PurePosixPath, Fraction]
type BookStatsDict = dict[PurePosixPath, BookReadStats]
type BookMetadataDict = dict[PurePosixPath, BookMetadata]


def get_progress(
    progress_file: Path, book_info: BookMetadataDict
) -> OverallProgressDict:
    """For each book, get reading progress as a percentage."""
    progress = {}
    no_match = []

    # the progress_file stores the file paths in lowercased form.
    # we want to use title case everywhere, so use this to map it back to book_info keys
    names_lower = {str(file).lower(): file for file in book_info}

    progress_data = progress_file.read_bytes()

    soup = BeautifulSoup(progress_data, "xml")
    assert soup.map is not None, f"invalid progress_data {soup}"
    for entry in soup.map.find_all("string"):
        if not isinstance(entry, bs4.Tag):
            continue
        if not entry.string:
            continue
        file_path_raw = entry.attrs['name']
        assert isinstance(file_path_raw, str)
        file_path_raw = names_lower.get(file_path_raw, file_path_raw)
        file = PurePosixPath(file_path_raw)
        if match := PROGRESS_RE.match(entry.string):
            val = Fraction(match['percentage'])
            if old := progress.get(file):
                # just to be extra sure
                logging.warning(f"duplicate! {file=!r}, {old=!r}, {val=!r}")
            progress[file] = val
        else:
            no_match.append((file, entry))
    if no_match:
        logging.warning(f"No progress found for {no_match}")
    return progress


def get_daily_progress(
    db_con: sqlite3.Connection,
) -> BookStatsDict:
    cur = db_con.execute("SELECT * FROM 'statistics'")
    stats: BookStatsDict = {}
    for row in cur.fetchall():
        file = PurePosixPath(row['filename'])
        reading_time = td(milliseconds=row['usedTime'])
        all_stats = [DailyStats.from_str(line) for line in row['dates'].splitlines()]
        book_stats = BookReadStats(reading_time, row['readWords'], all_stats)
        if old := stats.get(file):
            # just to be extra sure
            logging.warning(f"duplicate! {file=!r}, {old=!r}, {book_stats=!r}")
        stats[file] = book_stats
    cur.close()
    return stats


def get_book_info(db_con: sqlite3.Connection) -> BookMetadataDict:
    cur = db_con.execute("""SELECT t.filename,
            COALESCE(b.book, t.book) as book,
            COALESCE(b.author, t.author) as author
        FROM 'books' b RIGHT JOIN 'tmpbooks' t ON b.filename = t.filename;""")
    metadata = {}
    for row in cur.fetchall():
        file = PurePosixPath(row['filename'])
        title = row['book']
        author = row['author']

        metadata[file] = BookMetadata(title, author)
    return metadata


def get_unique_books(
    books: BookMetadataDict, progress: OverallProgressDict
) -> BookMetadataDict:
    """There may be multiple files corresponding to a single book.
    Use similarity matching on title to group the books. Then use progress%
    to trim out the older/unread version.
    """

    matcher = difflib.SequenceMatcher()
    CUTOFF = 0.8

    def normalize_title(title: str) -> list[str]:
        """Removes some common terms and splits by whitespace."""
        if idx := title.find(' by ') > 1:
            title = title[:idx]
        title = title.replace("z-lib.org", "")
        title = title.replace(".epub", "")
        # TODO: Remove punctuation?
        return title.split()

    def get_similar(
        other_books: BookMetadataDict,
        file: PurePosixPath,
        info: BookMetadata,
        field_name: Literal['title'] | Literal['author'],
    ):
        matcher.set_seq1(getattr(info, field_name))
        similar = set()
        for other_file, other_book in other_books.items():
            if other_file == file:
                continue
            matcher.set_seq2(getattr(other_book, field_name))
            if matcher.ratio() > CUTOFF:
                similar.add(other_file)
        return similar

    # sort of abusing dataclasses here. storing list of strings in title
    # difflib can accept list of strings
    rem_books = {
        file: BookMetadata(normalize_title(book.title), book.author)  # type: ignore
        for file, book in books.items()
    }
    groups: list[set[PosixPath]] = []

    for file in books.keys():
        if (info := rem_books.get(file)) is None:
            continue
        similar_titles = get_similar(rem_books, file, info, 'title')
        if info.author:
            similar_authors = get_similar(rem_books, file, info, 'author')
            # print(file.name, similar_titles, similar_authors)
            dups = similar_titles & similar_authors
        else:
            dups = similar_titles
        dups.add(file)
        # print(file.name, dups)
        groups.append(dups)

        # in one, but not the other
        # extra = similar_titles.symmetric_difference(similar_authors)
        # if extra:
        #     print("extra", file.name, extra)

        # remove the group from further processing
        for dup in dups:
            del rem_books[dup]

    # debug assert
    for (g1, g2) in combinations(groups, r=2):
        if not g1.isdisjoint(g2):
            raise Exception(f"Book(s) {g1 & g2} in both groups", g1, g2)

    logging.info(f"got {len(groups)} unique books from total {len(books)} books")

    unique_books = {}
    for group in groups:
        if len(group) > 1:
            max_progress = max(group, key=lambda file: progress.get(file, 0))
            # verbose info
            prog = round(float(progress.get(max_progress, 0)), 2)
            d = {k.name: progress.get(k, 0) for k in group}
            logging.debug(f"picked {max_progress.name} at {prog} from {d}")
        else:
            max_progress = next(iter(group))
        if max_progress not in progress:
            logging.debug(
                f"removing book {max_progress} because it is missing from progress file"
            )
            continue
        unique_books[max_progress] = books[max_progress]
    return unique_books


def parse_manual_progress(file: Path) -> dict[BookMetadata, list[DailyStats]]:
    if not file.exists():
        return {}
    data = tomllib.loads(file.read_text())
    output = {}
    # could probably use pydantic for this, but meh
    for book in data.get("books", []):
        bmd = BookMetadata(book['title'], book['author'])
        daily_stats: list[DailyStats] = []
        output[bmd] = daily_stats
        for session in book['sessions']:
            if 'day' in session:  # single date
                assert isinstance(session['day'], date)
                ds = DailyStats(
                    day=session['day'],
                    num_words=session.get("num_words", 0),
                    reading_time=td(seconds=session['reading_time']),
                )
                daily_stats.append(ds)
            else:  # range of dates
                assert isinstance(session['start'], date)
                assert isinstance(session['end'], date)
                total_time = session['reading_time']
                total_words = session.get('num_words', 0)
                end = session['end'] + td(days=1)
                num_days = (end - session['start']).days
                time_per_day = td(seconds=total_time / num_days)
                words_per_day = total_words / num_days
                day = session['start']
                while day != end:
                    ds = DailyStats(
                        day=day, num_words=words_per_day, reading_time=time_per_day
                    )
                    daily_stats.append(ds)
                    day = day + td(days=1)

        for (day1, day2) in combinations(daily_stats, r=2):
            assert day1.day != day2.day, f"overlapping days found! {day1} {day2}"
    return output


def make_con(db_file):
    con = sqlite3.connect(db_file)
    con.row_factory = sqlite3.Row
    return con


def process_backup(ignore_matcher, db_file: Path, progress_file: Path):
    con = make_con(db_file)
    daily_progress = get_daily_progress(con)
    logging.debug(pretty_repr(daily_progress))
    book_info = get_book_info(con)
    logging.debug(pretty_repr(book_info))
    progress = get_progress(progress_file, book_info)
    progress = {
        path: prog for path, prog in progress.items() if not ignore_matcher(path)
    }

    unique_books = get_unique_books(book_info, progress)
    logging.debug(pretty_repr({f.name: i for f, i in unique_books.items()}))
    return daily_progress, unique_books



def get_reading_time(
    data_files: list[tuple[Path, Path]],
    manual_progress_file: Path,
    ignore_matcher: IgnoreMatcher = DEFAULT_IGNORE_MATCHER,
) -> dict[date, td]:
    # file -> archive ->
    processed = defaultdict(dict)
    reading_time_by_date = defaultdict(td)
    total_reading_time_by_date = defaultdict(td)
    results = starmap(partial(process_backup, ignore_matcher), data_files)
    for (db_file, _), (daily_progress, unique_books) in zip(data_files, results):
        for file, stats in daily_progress.items():
            for day in stats.daily_stats:
                if day in processed[file]:
                    logging.debug(f"Cross-archive duplicate! '{file}' {day} first seen in '{db_file}'")
                    continue
                total_reading_time_by_date[day.day] += day.reading_time
                if file not in unique_books:
                    continue
                reading_time_by_date[day.day] += day.reading_time
                processed[file][day] = db_file

    # add values from manual progress entries
    for stats in parse_manual_progress(manual_progress_file).values():
        for day in stats:
            reading_time_by_date[day.day] += day.reading_time

    reading_time_by_date = {
        d: reading_time_by_date[d] for d in sorted(reading_time_by_date)
    }

    stats = False
    if stats:
        total_read_time = sum(reading_time_by_date.values(), start=td(0))
        logging.info(total_read_time)
        logging.debug(sum(total_reading_time_by_date.values(), start=td(0)))
    return reading_time_by_date


def render_graph(reading_time_by_date, out_path: Path):
    from render_html import create_graph

    t = create_graph(reading_time_by_date)
    out_path.write_text(t)


def get_data_files(data_dir: Path):
    assert data_dir.exists()

    namesfile = data_dir / "_names.list"
    db_file, progress_file = None, None
    for i, line in enumerate(namesfile.read_text().strip().splitlines(), 1):
        if line.endswith("positions10.xml"):
            progress_file = data_dir / f"{i}.tag"
            assert progress_file.exists()
        elif line.endswith("mrbooks.db"):
            db_file = data_dir / f"{i}.tag"
            assert db_file.exists()
    assert db_file
    assert progress_file
    return db_file, progress_file


def extract_data_archive(archive_path: Path) -> Path:
    zf = zipfile.ZipFile(archive_path)
    extract_path = Path(tempfile.mkdtemp(prefix=f"moon_reader_{archive_path.stem}"))
    zf.extractall(extract_path)
    inner_dir = extract_path / "com.flyersoft.moonreaderp"
    for f in inner_dir.iterdir():
        f.rename(extract_path / f.name)
    inner_dir.rmdir()
    return extract_path


def get_ignore_matcher(data_dir: Path, ignorefile: str) -> IgnoreMatcher:
    options = (
        Path(ignorefile),
        data_dir / ignorefile,
        data_dir.parent / ignorefile,
        Path.cwd() / ignorefile,
    )
    for path in options:
        if path.exists():
            return parse_gitignore(path, MR_ATTACH_DIR)

    # no matches
    if ignorefile != IGNOREFILE_NAME:
        # user has provided a custom ignorefile. Halt if not found
        raise Exception(
            f"Could not locate {ignorefile=} in any of " + ", ".join(map(str, options))
        )
    return DEFAULT_IGNORE_MATCHER


def main():
    parser = argparse.ArgumentParser("moonm")
    parser.add_argument("--stats", action="store_true", default=False)
    parser.add_argument("--loglevel", default="INFO", choices=logging._nameToLevel)
    data_p = parser.add_mutually_exclusive_group(required=True)
    data_p.add_argument("--data-dir", type=Path)
    data_p.add_argument("--archive-path", type=Path, nargs="+")
    parser.add_argument(
        "--ignorefile",
        help="Path/name of ignore file. It should follow the gitignore file format.",
        default=IGNOREFILE_NAME,
    )
    parser.add_argument(
        "--manual-progress-file", type=Path, default=Path("data/manual.toml")
    )
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--action", default="json", choices=["json", "html"])
    args = parser.parse_args()
    logging.root.setLevel(logging._nameToLevel[args.loglevel])

    if args.archive_path:
        data_dirs = [extract_data_archive(p) for p in args.archive_path]
    else:
        data_dirs = [args.data_dir]
    ignore_matcher = get_ignore_matcher(data_dirs[0], args.ignorefile)

    data_files = [get_data_files(data_dir) for data_dir in data_dirs]

    reading_time_by_date = get_reading_time(
        data_files, args.manual_progress_file, ignore_matcher
    )
    if args.action == "json":
        with open(args.outfile, "w") as f:
            json.dump(
                {"stats": {
                    d.isoformat(): v.total_seconds()
                    for d, v in reading_time_by_date.items()
                }},
                f,
                indent=2,
            )
        logging.info(f"Wrote json output to {str(args.outfile)!r}")
    elif args.action == "html":
        render_graph(reading_time_by_date, args.outfile)

    if args.archive_path:
        # clean up
        for data_dir in data_dirs:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
