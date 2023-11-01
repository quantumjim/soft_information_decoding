import warnings
import datetime

from .IQ_data import *
from .UnionFind import *
from .Hardware import *
from .PyMatching import *


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{datetime_str} Warning: {message}. IN FILE: {filename}, LINE: {lineno}")


# Override the default showwarning function
warnings.showwarning = custom_showwarning
