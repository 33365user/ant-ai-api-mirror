import string
from enum import Enum
import logging as log
from datetime import timedelta
import datetime
import re
import traceback
from functools import wraps, cache
import sys
import os
import random
from io import TextIOWrapper
from typing import Callable
from queue import Queue, PriorityQueue

import pandas as pd

from params import Params
from datetime import datetime as dt
import hashlib as hl

def check_relative_path_exists(relative_path):
    absolute_path = os.path.abspath(relative_path)
    return os.path.exists(absolute_path)


def ensure_folder_exists(relative_path):
    absolute_path = os.path.abspath(relative_path)
    os.makedirs(absolute_path, exist_ok=True)
    return check_relative_path_exists(relative_path)


def copy_doc(copy_func: Callable) -> Callable:
    """Use Example: copy_doc(self.copy_func)(self.func) or used as deco"""

    def wrapper(func: Callable) -> Callable:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapper


def generate_random_string(length=20):
    characters = string.ascii_letters + string.digits
    random_filename = ''.join(random.choice(characters) for _ in range(length))
    return random_filename


def search_queue(queue: Queue | PriorityQueue, target_identity):
    # copy to new queue that can be depleted
    found_requests = [e for e in queue.queue if e.identity == target_identity]
    return found_requests


def change_type(data:object, totype:type, default=0, **kwargs):
    """
    Primitively attempts to convert a parameter to a sepcified type `totype`, may fail, so returns `default` in that case.
    :param data: any object or value
    :param totype: any base type or inherits standard conversion methodology
    :param default:default value to assign on type error or equivalent - fail-proof.
    :return: data in type `totype` or `default` if fail.
    """
    d = data
    try:
        d = totype(d, **kwargs)
    except BaseException:
        d = default
    return d


class ClassificationRequest:
    identity: str
    filepath: str  # relative or absolute, prefer absolute
    status: int = 0  # 0 = received but un-actioned, 1 = In Progress, 2 = Complete, 3 = Rejected
    createdat: datetime.datetime = datetime.datetime.now()
    details: dict = {}
    checkedat: datetime.datetime = datetime.datetime.now()
    # user-generated data
    # "occurenceID" (autogenned, hashed), "basisOfRecord", "decimalLatitude", "decimalLongitude", "geodeticDatum",
    #                    "coordinateUncertaintyInMeters"
    basisOfRecord: str = "HumanObservation"
    decimalLatitude: float = 0.0
    decimalLongitude: float = 0.0
    geodeticDatum: str = "WGS84"
    coordinateUncertaintyInMeters: float | int = 30

    def __init__(self, identity, filepath, decimalLatitude=0.0, decimalLongitude=0.0, geodeticDatum="WGS84",
                 coordinateUncertaintyInMeters=30):
        self.identity = identity
        self.filepath = filepath
        # params used by program (metadata)
        self.status = 0
        self.createdat = datetime.datetime.now()
        self.details = dict(notes="Empty")
        self.checkedat = datetime.datetime.now()
        # user-generated data
        self.basisOfRecord = "HumanObservation"
        self.decimalLatitude = decimalLatitude
        self.decimalLongitude = decimalLongitude
        self.geodeticDatum = geodeticDatum
        self.coordinateUncertaintyInMeters = coordinateUncertaintyInMeters

    @property
    def occurenceID(self):
        """
        ALA Integration Unique OccurrenceID
        Note: Generated on-demand; in a static context, guaranteed to be unique, but two references to this property of this object may not hold the same value!
        :return:
        """
        hashn = hl.sha256(self.identity.encode('utf-8'))
        hashn.update(self.createdat.isoformat().encode('utf-8'))
        return hashn

    def __eq__(self, other):
        if type(other) == ClassificationRequest:
            return self.identity == other.identity
        elif type(other) == str:
            return self.identity == other
        else:
            return self.identity == other

    @property
    def out_of_date(self):
        return (dt.now() - self.createdat) > Params.QUEUE_TIMEOUT

    def get_csvlines(self) -> list[dict]:
        lines = []
        # as there may be multiple per image
        for i, (sname, score, rois) in enumerate(
                zip(self.details['species_name'], self.details['score'], self.details['regions_of_interest'])):
            l = self.__get_csv_line__(sname, score, rois, i)
            lines.append(l)
        return lines

    def __get_csv_line__(self, species_name, score, rois, n=0):
        occ = self.occurenceID
        occ.update(str(n).encode('utf-8'))

        ## CSV_COLUMNS = ["occurenceID", "basisOfRecord", "decimalLatitude", "decimalLongitude", "geodeticDatum",
        ##                  "coordinateUncertaintyInMeters", "eventDate", "uri", "identity", "scientificName",
        #                   "score", "regionsOfInterest", "instanceN"]

        vals = [str(occ.hexdigest()), self.basisOfRecord, self.decimalLatitude, self.decimalLongitude,
                self.geodeticDatum, self.coordinateUncertaintyInMeters, self.createdat.isoformat(),
                f"/uploads/{self.identity}", self.identity, species_name, score, rois, n]
        csvline = dict([(c, vals[i]) for i, c in enumerate(Params.CSV_COLUMNS)])
        return csvline

    def write_to_csv(self, filepath):
        print(f"{filepath}, {type(filepath)=}")
        lines = self.get_csvlines()
        df: pd.DataFrame = pd.read_csv(filepath, parse_dates=True)
        for l in lines:
            df = df._append(l, ignore_index=True)
        df.to_csv(filepath, index=False)


def get_week_boundaries(date):
    """
    Finds the start and end of week for the date `date`. Returns in that order, as a tuple.
    :param date:
    :return:
    """
    # Find the start of the week (Monday)
    start_of_week = date - timedelta(days=date.weekday())
    # Find the end of the week (Sunday)
    end_of_week = start_of_week + timedelta(days=6)
    # specify as much of the range as possible
    end_of_week = end_of_week.replace(hour=23, minute=59, second=59)
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0)

    return start_of_week, end_of_week


def get_month_boundaries(date):
    # Find the first day of the month
    first_day_of_month = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Find the last day of the month
    next_month = first_day_of_month + timedelta(days=32)
    last_day_of_month = (next_month.replace(day=1) - timedelta(days=1)).replace(hour=23, minute=59, second=59,
                                                                                microsecond=999999)

    return first_day_of_month, last_day_of_month


def get_quarter_boundaries(date):
    quarter = (date.month - 1) // 3 + 1
    first_day_of_quarter = dt(date.year, 3 * (quarter - 1) + 1, 1, 0, 0, 0, 0)
    last_day_of_quarter = dt(date.year, 3 * quarter, 1, 0, 0, 0, 0) + timedelta(days=30)
    last_day_of_quarter = min(last_day_of_quarter, dt(date.year, 12, 31, 23, 59, 59, 999999))
    return first_day_of_quarter, last_day_of_quarter


def FormHTTPMessage(status: int, message: str, other: dict):
    o = {'status': str(status), 'message': message}
    o.update(other)
    return o
