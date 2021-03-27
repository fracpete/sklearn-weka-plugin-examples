import os
import tempfile


def get_data_dir():
    """
    Returns the data directory.

    :return: the data directory
    :rtype: str
    """
    rootdir = os.path.dirname(__file__)
    libdir = rootdir + os.sep + "data"
    return libdir


def print_title(title):
    """
    Prints the title underlined.

    :param title: the title to print
    :type title: str
    """

    print("\n" + title)
    print("=" * len(title))


def print_info(info):
    """
    Prints the info.

    :param info: the info to print
    :type info: str
    """

    print("\n" + info)


def get_tmp_dir():
    """
    Returns the tmp directory.

    :return: the tmp directory
    """
    return tempfile.gettempdir()
