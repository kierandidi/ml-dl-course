#!/usr/bin/env python3
"""Deprecated: Week 2 lectures (days 6–10) are now authored in content_dayNN.py
and rendered by generate_lectures.py.

Days 9–10 used to be generated here from thin inline markdown. They have been
upgraded to full content modules (content_day09.py, content_day10.py) on par with
Week 1 and are emitted by generate_lectures.py alongside every other day. This
shim is kept only so existing instructions/scripts that call it do not error; it
intentionally writes nothing.
"""
from __future__ import annotations


def main() -> None:
    print(
        "generate_lectures_w2.py is deprecated and does nothing.\n"
        "Days 9–10 are now produced by scripts/generate_lectures.py "
        "(content_day09.py / content_day10.py). Run that instead."
    )


if __name__ == "__main__":
    main()
