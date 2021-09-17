#!/usr/bin/python

from typing import (
    Iterable,
    List,
    Dict,
    Any
)


def make_dict(map_name: Iterable[Dict[str, int]]) -> Any:
    print(map_name)


def sample_2() -> Dict[str, Dict[str, int]]:
    return {
        "DictName": {"age": 28, "age1": 32, "age2": 50}
    }


def iterate_elements() -> Iterable[int]:
    return [x for x in range(10)]


def sample_1() -> List[str]:
    return [x for x in "hello world"]


def main():
    # make_dict(map_name=[{"age": 28, "age1": 32, "age2": 50}])
    print(sample_2())


if __name__ == "__main__":
    main()
