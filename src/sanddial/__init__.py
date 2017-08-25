"""Initialize the sand dial clock program, by setting up signal handlers
and performing reading/parsing of command line arguments before passing
off control of the program to the other modules.
"""

import signal
import sys
from sanddial import err
from sanddial import clock


def graceful_exit(sig, frame):
    """Handle Ctrl+C signal to cancel execution without a Python stack trace"""
    err.warn("\nExiting; oh how time flies. " +
             "Caught signal " + str(sig) + " in frame " + str(frame) + ".")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
err.log('Press Ctrl+C when done with clock.')

clock.main()

# if should_turn:
#     minute += 1
# if minute >= 60:
#     hour += 1
#     minute = 0
# if hour >= 24:
#     hour = 0
#     servo.turn()