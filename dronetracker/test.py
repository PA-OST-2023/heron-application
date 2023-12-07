# SuperFastPython.com
# example of wait for with a condition
from time import sleep
from random import random
from threading import Thread
from threading import Condition
 
# target function
def task(condition, work_list, i):
    # acquire the condition
    with condition:
        # block for a moment
        value = random()
        sleep(value)
        # add work to the list
        work_list.append(value)
        print(f'Thread {i} added {value}')
        print(len(work_list))
        # notify the waiting thread
        condition.notify_all()
 
def hans(condition):

    while True:
        sleep(0.1)
        with condition:
            # wait to be notified
            if(not condition.wait_for(lambda : len(work_list)%5==0, timeout=5)):
                continue
        #     condition.wait()
            print('condiion True')

            print(f'Done, got: {work_list}')



# create a condition
condition = Condition()
# define work list
work_list = list()
# start a bunch of threads that will add work to the list
arbeiter = Thread(target=hans, args=(condition,))
arbeiter.start()
for i in range(19):
    sleep(random())
    worker = Thread(target=task, args=(condition, work_list, i))
    worker.start()
print('What')
# wait for all threads to add their work to the list
# with condition:
#     # wait to be notified
#     condition.wait_for(lambda : len(work_list)>=5)
# #     condition.wait()
#     print('condiion True')
# # sleep(2)
# print(f'Done, got: {work_list}')
