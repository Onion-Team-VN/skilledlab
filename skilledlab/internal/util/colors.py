"""
Console colors
"""
from enum import Enum

_ANSI_CODES = dict(
    normal=0,
    bold=1,
    light=2,  # - PyCharm/Jupyter

    italic=3,  # - PyCharm/Jupyter
    underline=4,

    highlight=7,  # Changes background in PyCharm/Terminal

    # Colors
    black=30,
    red=31,
    green=32,
    orange=33,
    blue=34,
    purple=35,
    cyan=36,
    white=37,
    link=4
)

ANSI_RESET = "\33[0m"

class StyleCode(Enum):
    r"""
    This is the base class for different style enumerations
    """

    def ansi(self):
        if self.value is None:
            return f"\33[{_ANSI_CODES['normal']}m"
        elif type(self.value) == str:
            return f"\33[{_ANSI_CODES[self.value]}m"
        elif type(self.value) == list:
            return ''.join([f"\33[{_ANSI_CODES[v]}m" for v in self.value])
        else:
            assert False

def _test():
    for i in [0, 38, 48]:
        for j in [5]:
            for k in range(16):
                print("\33[{};{};{}m{:02d},{},{:03d}\33[0m\t".format(i, j, k, i, j, k),
                      end='')
                if (k + 1) % 6 == 0:
                    print("")
            print("")

    for i in range(0, 128):
        print(f"\33[{i}m{i :03d}\33[0m ", end='')
        if (i + 1) % 10 == 0:
            print("")

    print()

    print("▁▂▃▄▅▆▇█")
    print("▁▂▃▄▅▆▇█")

if __name__ == "__main__":
    _test()

