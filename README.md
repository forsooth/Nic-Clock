# SandDial: Hourglass Based OpenCV Clock
Ah, the good old days of 200 BCE, when the primary means of telling time were sundials during the day, candles at night, and hourglasses if finer granularity was needed. Well in an unprecedented breakthrough in timekeeping technology, we present to you the time-telling device which combines the reliability of a dial with the granularity of an hourglass: the SandDial!

What is a SandDial, you may ask? Well, it's magnificent! It's astounding! It's an hourglass strapped to a servo, overseen by a Raspberry Pi camera, and turned automatically through the magic of OpenCV.

# Usage
Run the program using a Python 3 interpreter with an invocation (depending on your operating system and configurations) similar to:
```
python3 src/run.py
```

Alternatively, write your own frontend interface to the SandDial, by importing the `sanddial` package defined in `src/sanddial`.

# Arguments

# Expected Behavior
If starting up in the standard camera-servo-processor mode, once the user has confirmed to the program that physical setup is complete, the program is expected to take around 5 seconds to configure the camera and read some basic data from the camera sensor to determine the optimal image settings; therefore, do not move the initial setup during this period, and ensure that it reflects a consistent image-taking environment.

Once this is complete, a matplotlib window should open, containing nothing of interest. After a short period of image processing, the plot should be filled with processed images, and the current time should be displayed in the console. These plots will auto-update as the program progresses. The plot window should *not* be closed, or behavior is undefined.

In the case that a minute has passed and the servo must be triggered, the program's regular duties will pause while this transaction takes place.

# Architecture
SandDial is contained in a Python package named `sanddial`, which can be found at `src/sanddial/`. In addition, a small script `src/run.py` is provided as an example use of the package. In terms of outward-facing interface, the package is about as simple as possible: one class (SandDial) is defined, and this one class has a single method (run). As can be seen from the `src/run.py` script, the only user action required is to import the package, create a SandDial clock object, and begin running the clock.

Internally, things get a fair bit more complex. The SandDial object interfaces with three entities: the camera, the servo, and the image processor.

### Camera
The camera is the first component to be initialized by the SandDial. when initialized, it pauses the program for five seconds while it auto-configures its white balance, then freezes that white balance. Once configured, the camera can be queried repeatedly for a new frame, which it should happily provide.

### Servo
The motor controller has a simple task of rotating 180 degrees when requested, and rotating 90 degrees at the beginning and end of execution.

### Image Processor
The image processor is comprised of a half dozen different image operations which, when combined give back a contoured image of our hourglass in which sand can be easily detected. It can do so, and saves the height and width of the detected sand.

# PyLint
This project was combed with PyLint; an `src/.pylintrc` file exists, and so running `pylint sanddial` from within the `src` folder should give perfect results, assuming all modules are installed, etc. 

Changes from the default PyLint configuration are minimal, but for the sake of transparency are:
* allowing a module name exception for the C code extensions in the cv2 package
* shrinking the maximum line length from 100 to 80
* allowing several short but meaningful variable names.

# Dependencies


# Author
Matt Asnes, Summer 2017
