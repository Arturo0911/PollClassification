#!/usr/bin/python

from typing import Iterable


def iterate_elements() -> Iterable[int]:
    return [x for x in range(10)]


def main():
    for x in iterate_elements():
        print(x)


if __name__ == "__main__":
    main()
