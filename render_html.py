"""
The 'templates' directory contains two Jinja2 templates for rendering the
graph:

* `index.html` - the skeleton which only loads the CSS files, and then includes
  the output of the second template:
* `graph.html` - this is the template which actually renders a graph.

This module is responsible for preparing and rendering the templates.
"""

import datetime
from dataclasses import dataclass

from jinja2 import Environment
import jinja2

import dateutils

# GridCell = namedtuple('GridCell', ['date', 'contributions'])


@dataclass
class GridCell:
    date: datetime.date
    contributions: datetime.timedelta

    def tooltip_text(self):
        """
        Returns the tooltip text for a cell.
        """
        if self.contributions.total_seconds == 0:
            count = "No read"
        else:
            # TODO: humanize
            count = str(self.contributions)
        date_str = dateutils.display_date(self.date)
        return "<strong>%s</strong> on %s" % (count, date_str)


def create_graph(contributions: dict[datetime.date, datetime.timedelta]):
    """
    Prepare the `index.html` template.
    """
    graphs = []

    graph = {
        "data": gridify_contributions(contributions),
        "cell_class": _cell_class(contributions.values()),
        "sum": sum(
            contributions.values(), start=next(iter(contributions.values())).__class__()
        ),
        "repo_name": "Reading stats",
    }

    graph["last_date"] = (
        [""] + sorted([key for key, v in contributions.items() if v])
    )[-1]

    graphs.append(graph)
    env = Environment(loader=jinja2.FileSystemLoader("templates"))

    env.filters['tooltip'] = GridCell.tooltip_text
    env.filters['display_date'] = dateutils.display_date
    env.filters['elapsed_time'] = dateutils.elapsed_time

    template = env.get_template("index.html")

    weekdays = dateutils.WEEKDAY_INITIALS
    for idx in [0, 2, 4, 6]:
        weekdays[idx] = ""

    months = [
        cell.date.strftime("%b") for cell in gridify_contributions(contributions)[0]
    ]
    months = filter_months(months)

    return template.render(
        graphs=graphs,
        today=dateutils.today(),
        start=dateutils.start(),
        weekdays=weekdays,
        months=months,
    )


def gridify_contributions(contributions):
    """
    The contributions graph has seven rows (one for each day of the week).
    It spans a year. Given a dict of date/value pairs, rearrange these results
    into seven rows of "cells", where each cell records a date and a value.
    """
    start = dateutils.start()
    today = dateutils.today()
    print(start)

    graph_entries = []

    # The first row is a Sunday, so go back to the last Sunday before the start
    if start.weekday() == 6:
        first_date = start
    else:
        first_date = start - datetime.timedelta(start.weekday() + 1 % 7)
    next_date = first_date

    first_row_dates = [first_date]
    while (next_date <= today) and (next_date + datetime.timedelta(7) <= today):
        next_date += datetime.timedelta(7)
        first_row_dates.append(next_date)

    # Now get contribution counts for each of these dates, and save the row
    first_row = [GridCell(date, contributions[date]) for date in first_row_dates]
    graph_entries.append(first_row)

    # For each subsequent day of the week, use the first row as a model: add
    # the appropriate number of days and count the contributions
    for i in range(1, 7):
        row_dates = [day + datetime.timedelta(i) for day in first_row_dates]
        next_row = [GridCell(date, contributions[date]) for date in row_dates]
        graph_entries.append(next_row)

    return graph_entries


def _cell_class(values):
    """
    Returns a function which determines how a cell is highlighted.
    """
    # TODO: Custom partitiions
    # quartiles = statistics.quantiles(values, n=5)
    td = datetime.timedelta
    quartiles = [td(0), td(minutes=30), td(hours=1), td(hours=2)]
    print(quartiles)

    def class_label(cell):
        # print(f"{cell=}")
        if cell.date > dateutils.today() or cell.date < dateutils.start():
            return "empty"
        elif cell.contributions == quartiles[0]:
            # print("zero")
            return "grad0"
        elif cell.contributions <= quartiles[1]:
            return "grad1"
        elif cell.contributions <= quartiles[2]:
            return "grad2"
        elif cell.contributions <= quartiles[3]:
            return "grad3"
        else:
            return "grad4"

    return class_label


def filter_months(months):
    """
    We only want to print each month heading once, over the first column
    which contains days only from that month. This function filters a list of
    months so that only the first unique month heading is shown.
    """
    for idx in reversed(range(len(months))):
        if months[idx] == months[idx - 1]:
            months[idx] = ""

    # If the same month heading appears at the beginning and end of the year,
    # then only show it at the end of the year
    if months.count(months[0]) > 1:
        months[0] = ""
    if months.count(months[-1]) > 1:
        months[-1] = ""

    # Since each month takes up cells, we delete an empty space for each month
    # heading
    indices = [idx for idx, month in enumerate(months) if month]
    for idx in reversed(indices):
        if idx != len(months) - 1:
            del months[idx + 1]

    return months
