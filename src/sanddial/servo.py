import wiringpi
from time import sleep

GPIO_PIN = 18


class Servo():
    def __init__(self):
        # use 'GPIO naming'
        wiringpi.wiringPiSetupGpio()

        # set #18 to be a PWM output
        wiringpi.pinMode(18, wiringpi.GPIO.PWM_OUTPUT)

        # set the PWM mode to milliseconds stype
        wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)

        # divide down clock
        wiringpi.pwmSetClock(192)
        wiringpi.pwmSetRange(2000)

        self.pulse = 135
        self.zero_pulse = 48
        self.reset_pulse = 135
        self.flipped_pulse = 225
        self.reset()

    def flip(self):
        if self.pulse > self.zero_pulse:
            wiringpi.pwmWrite(GPIO_PIN, self.zero_pulse)
            self.pulse = self.zero_pulse
        elif self.pulse < self.flipped_pulse:
            wiringpi.pwmWrite(GPIO_PIN, self.flipped_pulse)
            self.pulse = self.flipped_pulse
        sleep(2)

    def reset(self):
        wiringpi.pwmWrite(GPIO_PIN, self.reset_pulse)
        self.pulse = self.reset_pulse

