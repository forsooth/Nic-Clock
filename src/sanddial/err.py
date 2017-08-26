"""This module implements several functions for printing to standard error
in a command-line environment using ANSI 256 color codes.
"""
import sys
from sanddial import colors

NL = '\n'

def err(msg=""):
    """Prints a fatal error in red, then exits the program."""
    sys.stderr.write(colors.RED + str(msg) + colors.NONE + NL)
    sys.exit(1)


def warn(msg=""):
    """Prints a warning in yellow."""
    sys.stderr.write(colors.YELLOW + str(msg) + colors.NONE + NL)


def log(msg=""):
    """Prints an informational message in blue."""
    sys.stderr.write(colors.BLUE + str(msg) + colors.NONE + NL)

def success(msg=""):
    """Prints a notification of a success in green."""
    sys.stderr.write(colors.GREEN + str(msg) + colors.NONE + NL)
