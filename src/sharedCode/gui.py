import datetime
import time
import sys


class SimpleProgressCounter:
    def __init__(self, max=100, caption=None):
        self.state = 0
        self.max = max
        self._time_progress_triggered_first_time = None
        self._displayed_once = False

        self._suffix = ""

        if caption is not None:
            self._suffix = caption + '   ...   '
        self.value = 'Jobs done 0/{}'.format(self.max)

    def display(self):
        if not self._displayed_once:
            print("", end='\n')

            self._displayed_once = True

        print(self._suffix + self.value, end='\r')
        sys.stdout.flush()

    def trigger_progress(self):
        self.state += 1
        text = 'Jobs done: {}/{}'.format(self.state, self.max)

        if self._time_progress_triggered_first_time is None:
            self._time_progress_triggered_first_time = time.time()

        else:
            avg = (time.time() - self._time_progress_triggered_first_time) / self.state
            estimated_remaining_time = (self.max - self.state) * avg
            text += '   Remaining time: {}'.format(str(datetime.timedelta(seconds=int(estimated_remaining_time))))

        self.value = text
        self.display()
