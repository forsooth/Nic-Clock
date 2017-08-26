"""Initialize the sand dial clock program, by setting up signal handlers
and performing reading/parsing of command line arguments before passing
off control of the program to the other modules.
"""
import signal
import sys
import datetime
from sanddial import err
from sanddial import imgproc
from sanddial import camera

# Number of frames for which we should double check that there is no sand
# in the image before turning the servo
MAX_STRIKES = 3


def graceful_exit(sig, frame):
    """Handle Ctrl+C signal to cancel execution without a Python stack trace"""
    err.warn("\nExiting; oh how time flies. " +
             "Caught signal " + str(sig) + " in frame " + str(frame) + ".")
    sys.exit(0)


class SandDial():
    """Module for orchestration of servo, camera, and image processing, as
    well as control for printing the current time to the terminal.
    """

    def __init__(self):
        """Set up package signal handler to SIGINT, initialize the camera
        and the image processor, and begin the clock procedure
        """
        # Print hello message
        err.success("Welcome to the SandDial time keeping system!")
        err.log("Press Ctrl+C when done with clock.")

        # Register response to interrupt signal
        signal.signal(signal.SIGINT, graceful_exit)

        err.log("Press Enter when physical setup of camera, computer, and " +
                "servo is complete and configuration of camera settings " +
                "(e.g. white balance) may commence.")
        input()

        # Initialize camera
        self.cam = camera.Camera()

        # Initialize image processor
        height, width = self.cam.get_width(), self.cam.get_height()
        self.processor = imgproc.ImageProcessor(width, height)

        self.strikes = 0

        # Grab current time from system default
        self.init_t = datetime.datetime.now()
        # If we're closer to the next minute, round up; we'll probably spend
        # a few seconds in the first round of the image processing anyway
        self.minute = self.init_t.minute + round(self.init_t.second / 100)
        self.hour = self.init_t.hour

        self.print_time()

    def print_time(self):
        """Output the current time to the terminal."""
        print(str(self.hour) + ':' + print(self.minute))

    def run(self):
        """Loop until the user quits with SIGINT, grabbing images from the
        camera, feeding them into the image processor, and moving the hourglass
        and printing out and updated time as required.
        """

        # Main run loop of the program; grab image, analyze it, act depending
        # on whether the image showed sand in the hourglass or not.
        while True:
            # Fetch a new frame from the camera
            frame = self.cam.get_frame()
            # Load the image into the image processor
            self.processor.load_img(frame)
            # Determine whether the servo should turn for this frame
            should_turn = self.processor.analyze()

            # When there is no sand left in the hourglass, we want to
            # make sure there really isn't any left by waiting for a couple
            # of images to corroborate the result
            if should_turn is True:
                self.strikes += 1

            # If we are in fact certain that there isn't any sand, then we
            # want to change our timer.
            if self.strikes >= MAX_STRIKES:
                self.strikes = 0
                self.minute += 1
                if self.minute >= 60:
                    self.minute = 0
                    self.hour += 1
                    if self.hour >= 24:
                        self.hour = 0
                self.print_time()
