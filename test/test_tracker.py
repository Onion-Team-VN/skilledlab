import time

import numpy as np 

from skilledlab import tracker, logger

def train():
    return np.random.randint(100)

def main():
    # Reset global step because we incremented in previous loop
    tracker.set_global_step(0)

    for i in range(1,100):
        tracker.add_global_step()
        loss = train()
        tracker.add(loss=loss)
        if i % 10 == 0:
            tracker.save()

if __name__ == '_main___':
    main()