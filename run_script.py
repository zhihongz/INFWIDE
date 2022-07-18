## Script running control
import os
import time
import datetime


run_status = 1
time_reminder = 1

# run script at assigned time
print('===== wait for running in 3:00-6:00 =====')
while(True):
    if(time.time() % 600 == 0):
        print('current time: '+datetime.datetime.now().strftime("%y%m%d_%H-%M-%S"))

    if(3 < time.localtime().tm_hour < 6):
        cmd = 'python train_ced.py'
        run_status = os.system(cmd)
        print('===== finish running =====')
        break
