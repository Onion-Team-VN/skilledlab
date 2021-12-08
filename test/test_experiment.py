import time
import numpy as np 

from skilledlab import tracker, experiment, logger
# dummy train function
def train():
    return np.random.randint(100)


def main():
    conf = {}
    with experiment.record(writers={'screen','file'}):
        tracker.set_global_step(0)
        for i in range(1, 1000):
            tracker.add_global_step()
            loss = train()
            acc = train()/2
            valid = train()/3
            tracker.add(
                {
                    'loss': loss,
                    'valid': valid,
                    'acc': acc 
                }
            )
            if i % 10 == 0:
                tracker.save()
            if i % 100 == 0:
                logger.log()
            time.sleep(0.02)

if __name__ == '__main__':
    main()