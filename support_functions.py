import json
import numpy as np
import serial
import serial.serialutil
import serial.tools.list_ports
import time
import threading


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that supports NumPy arrays.

    This class extends the standard JSONEncoder to handle NumPy arrays.
    When encountering a NumPy array in the object to be serialized, it converts
    the array to a Python list using the 'tolist()' method.
    """

    def default(self, o):
        """
        Override the default method of JSONEncoder.

        Parameters
        ----------
        o : object
            The object to be serialized.

        Returns
        -------
        JSON-serializable object
            The serialized version of the object.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def AsciiDecoder(b) -> str:
    """
    Decode key presses from the waitKey function in OpenCV.

    Parameters
    ----------
    b : int
        The keypress code received from the waitKey function.

    Returns
    -------
    str
        Returns the decoded character corresponding to the keypress.
        If the keypress code is '-1', returns "".
    """
    if b == '-1':
        return ""
    # bitmasks the last byte of b and returns decoded character
    return chr(b & 0xFF)