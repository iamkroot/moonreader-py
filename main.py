from collections import Counter, defaultdict
import difflib
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from datetime import timedelta as td
from fractions import Fraction
from itertools import combinations
from pathlib import Path, PosixPath, PurePosixPath
from pprint import pprint
from typing import Literal

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
        reading_time = td(milliseconds=int(match['time_ms']))
        words = int(match['words'])
        return DailyStats(day, reading_time, words)


@dataclass(slots=True, frozen=True)
class BookReadStats:
    reading_time: td
    read_words: int
    daily_stats: list[DailyStats]


@dataclass(slots=True)
class BookMetadata:
    title: str
    author: str


OverallProgressDict = dict[PurePosixPath, Fraction]
BookStatsDict = dict[PurePosixPath, BookReadStats]
BookMetadataDict = dict[PurePosixPath, BookMetadata]


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
        file = entry.attrs['name']
        if file_titlecase := names_lower.get(file):
            file = PurePosixPath(file_titlecase)
        else:
            file = PurePosixPath(file)
        if match := PROGRESS_RE.match(entry.string):
            val = Fraction(match['percentage'])
            if old := progress.get(file):
                # just to be extra sure
                print(f"duplicate! {file=}, {old=}, {val=}")
            progress[file] = val
        else:
            no_match.append(file)

    return progress


def get_daily_progress(
    db_con: sqlite3.Connection,
) -> BookStatsDict:
    cur = db_con.execute("SELECT * FROM 'statistics'")
    stats = {}
    for row in cur.fetchall():
        file = PurePosixPath(row['filename'])
        reading_time = td(milliseconds=row['usedTime'])
        all_stats = [DailyStats.from_str(line) for line in row['dates'].splitlines()]
        book_stats = BookReadStats(reading_time, row['readWords'], all_stats)
        if old := stats.get(file):
            # just to be extra sure
            print(f"duplicate! {file=}, {old=}, {book_stats=}")
        stats[file] = book_stats
    cur.close()
    return stats


def get_book_info(db_con: sqlite3.Connection) -> BookMetadataDict:
    cur = db_con.execute("SELECT * FROM 'tmpbooks'")
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

    print("got", len(groups), "unique books from total", len(books), "books")

    unique_books = {}
    for group in groups:
        if len(group) > 1:
            max_progress = max(group, key=lambda file: progress.get(file, 0))
            prog = round(float(progress[max_progress]), 2)
            print(
                f"picked {max_progress.name} at {prog} from",
                {k.name: progress.get(k, 0) for k in group},
            )
        else:
            max_progress = next(iter(group))
        unique_books[max_progress] = books[max_progress]
    return unique_books


def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    daily_progress = get_daily_progress(con)
    # pprint(daily_progress)

    book_info = get_book_info(con)
    # pprint(book_info)

    progress = get_progress(PROG_FILE, book_info)
    # pprint(progress)

    unique_books = get_unique_books(book_info, progress)
    pprint({f.name: i for f, i in unique_books.items()})
    reading_time_by_date = defaultdict(td)
    total_reading_time_by_date = defaultdict(td)
    for file, stats in daily_progress.items():
        for day in stats.daily_stats:
            total_reading_time_by_date[day.day] += day.reading_time
        if file not in unique_books:
            continue
        for day in stats.daily_stats:
            reading_time_by_date[day.day] += day.reading_time

    pprint({k: str(v) for k, v in reading_time_by_date.items()})
    print(sum(total_reading_time_by_date.values(), start=td(0)))
    print(sum(reading_time_by_date.values(), start=td(0)))

    from render_html import create_graph

    t = create_graph(reading_time_by_date)
    Path("temp.html").write_text(t)


if __name__ == "__main__":
    main()
