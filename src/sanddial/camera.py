"""Initialize camera settings and get OpenCV-ready images on request."""
from time import sleep
import numpy as np
import picamera
from sanddial import err

# Standard height and width values for pictures the camera should take,
# measured in pixels.
IMG_WIDTH =  480
IMG_HEIGHT = 640


class Camera():
    """Maintains an instance of the Raspberry Pi Camera library's camera
    object, and returns frames or frame information on request, formatted
    conveniently for OpenCV processing.
    """

    def __init__(self):
        """Initializes camera settings, and creates the camera object"""

        # frame size properties
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT

        # Note that the number of channels should not be changed, as OpenCV
        # immutably requires 3 channels (BGR) for color processing.
        self.channels = 3

        self.camera = picamera.PiCamera(resolution=(self.width, self.height))

        self.camera.iso = 100

        # We wait five seconds before setting the white balance so that
        # the camera has time to auto-adjust.
        err.log("Initializing camera white balance — sleeping for 5 seconds.")
        sleep(5)
        err.log("White balance initialized.")

        # After allowing the white balance to settle, we turn off its
        # ability to auto-adjust.
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        self.camera.awb_mode = 'off'

        self.gain = self.camera.awb_gains
        self.camera.awb_gains = self.gain

    def get_width(self):
        """Fetches the frame width."""
        return self.width

    def get_height(self):
        """Fetches the frame height."""
        return self.height

    def get_frame(self):
        """Fetches a new frame from the camera, and returns it as a
        numpy array in BGR format, i.e. the format that
        OpenCV creates and expects in order to perform image processing.
        """
        frame = np.empty((self.width * self.height * self.channels),
                         dtype=np.uint8)

        self.camera.capture(frame, 'bgr')
        frame = frame.reshape((self.height, self.width, self.channels))

        return frame
