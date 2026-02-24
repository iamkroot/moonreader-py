import datetime


def today():
    """
    Gets the current date. Wrapper function to make it easier to stub out in
    tests.
    """
    return datetime.date.today()


def get_weekday(weekday=0, date=today(), future_date=False) -> datetime.date:
    """Get the date for the given weekday for the week corresponding to date.
    Weekday = 0 means monday.
    """
    value = date + datetime.timedelta(weekday - date.weekday() % 7)
    return value + datetime.timedelta(7 * (future_date and date == value))


def start_exact():
    """
    Gets the date from one year ago.
    """
    return datetime.date(today().year - 1, today().month, today().day)


def start():
    """
    Gets the sunday from one year ago, which is the start of the contributions
    graph.
    """
    return get_weekday(6, start_exact())


def display_date(date: datetime.date):
    """
    Returns a long date string. Example output: "May 24, 2015".
    """
    return date.strftime("%B %d, %Y").replace(" 0", " ")


def previous_day(date: datetime.date):
    """
    Returns the previous day as a datetime.date object.
    """
    return date - datetime.timedelta(1)


def next_day(date: datetime.date):
    """
    Returns the next day as a datetime.date object.
    """
    return date + datetime.timedelta(1)


def elapsed_time(date: datetime.date):
    """
    Given a date in the past, return a human-readable string explaining how
    long ago it was.
    """
    if date > today():
        raise ValueError("Date {} is in the future, not the past".format(date))

    difference = (today() - date).days

    # I'm treating a month as ~30 days. This may be a little inaccurate in some
    # months, but it's good enough for our purposes.
    if difference == 1:
        return "a day ago"
    elif difference < 30:
        return "%d days ago" % difference
    elif difference < 30 * 2:
        return "a month ago"
    elif difference < 366:
        return "%d months ago" % (difference / 30)
    else:
        return "more than a year ago"


WEEKDAY_INITIALS = ['Su', 'M', 'Tu', 'W', 'Th', 'F', 'Sa']
