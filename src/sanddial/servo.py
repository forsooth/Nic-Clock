import wiringpi

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

        self.pulse = 100
        self.zero_pulse = 10
        self.reset_pulse = 102
        self.flipped_pulse = 195
        self.reset()

    def flip(self):
        if self.pulse > self.zero_pulse:
            wiringpi.pwmWrite(GPIO_PIN, self.zero_pulse)
        elif self.pulse < self.flipped_pulse:
            wiringpi.pwmWrite(GPIO_PIN, self.flipped_pulse)

    def reset(self):
        wiringpi.pwmWrite(GPIO_PIN, self.reset_pulse)
        self.pulse = self.reset_pulse

