from skilledlab import monitor, tracker


def main():
    import time

    for _ in monitor.loop(10):
        for n, v in monitor.mix(2, ('train', range(50)), ('valid', range(10))):
            time.sleep(0.05)
            # print(n, v)
            tracker.save({n: v})
        tracker.new_line()


if __name__ == '__main__':
    main()