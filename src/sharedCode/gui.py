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


__ask_user_for_provider_or_data_set_download_txt = \
"""
Persistence diagram provider does not exist! 
You have two options: 
1) Create persistence diagram provider from raw data
   this will take several hours and tda-toollkit has 
   to be installed properly.
2) Download precalculated persistence diagram provider.

Type 1 or 2 depending on your choice.
"""


def ask_user_for_provider_or_data_set_download():
    print(__ask_user_for_provider_or_data_set_download_txt)
    choice = input('-->')
    while str(choice) not in ['1', '2']:
        print("Choice has to be 1 or 2!")
        input()

    if choice == '1':
        return 'download_data_set'

    if choice == '2':
        return 'download_provider'
