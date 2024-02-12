import warnings
from datetime import datetime as dt

from .core import *
from .qubit_coordinates import *
from .metadata import *
from .calibration import *


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    datetime_str = dt.now().strftime("%H:%M:%S")
    print(f"{datetime_str} Warning: {message}. IN FILE: {filename}, LINE: {lineno}")


# Override the default showwarning function
warnings.showwarning = custom_showwarning
