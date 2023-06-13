# -*- coding: utf-8 -*-
"""
Utility for checking existing of different sets of data across multiple
flights with the Twin-Otter during the EUREC4A field campaign

Leif Denby 2020, GPL License
"""

from pathlib import Path
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored(s, level):
    if level == 'error':
        c = bcolors.FAIL
    elif level == 'success':
        c = bcolors.OKGREEN
    return "{}{}{}".format(c, s, bcolors.ENDC)


EXPECTED_PATHS = {
    "Brechtel": None,
    "CCN": "*.csv",
    "figures/": [
        "mphys_*.pdf",
        "height-time-with-legs.png",
        "flight{flight_num}_track_altitude.png",
        "Leg*_quicklook.png",
        "Profile*_skewt.png",
        ],
    "Grimm": None,
    "IMU": None,
    "MASIN": [
        "core_masin_*_1hz.nc",
    ],
    "Notes": None,
    "SMPS": None,
    "Video": None,
    "Photos": None,
    "flight{flight_num}-legs.csv": None,
}

FORMAT_INDENT = "{:40}"

flight_nums = sorted([
    int(p.name.replace('flight', '')) for p in Path().glob('flight*/')
])

def _print_path_status_for_all_flights(p_root, node):
    depth = len(str(p_root/node).split('/'))
    print(FORMAT_INDENT.format(" "*depth + node), end='')
    for flight_num in flight_nums:
        p_flight = Path('flight{}'.format(flight_num))
        p_node = p_flight/p_root
        g_pattern = node.format(flight_num=flight_num)
        n_matches = len(list(p_node.glob(g_pattern)))
        if n_matches > 0:
            print(colored("x", "success"), end=' ')
        else:
            print(colored("o", "error"), end=' ')
    print("")

def check_path(node, p_root):
    if type(node) == str:
        _print_path_status_for_all_flights(p_root, node)
    elif type(node) == dict:
        for (k, v) in node.items():
            _print_path_status_for_all_flights(p_root, k)
            if v is None:
                pass
            else:
                check_path(v, p_root/k)
    elif type(node) == list:
        for k in node:
            check_path(k, p_root)
    else:
        raise NotImplementedError(node)

def print_flight_nums():
    for n in range(len(str(flight_nums[-1]))):
        s = " "
        if n == 1:
            s = "         FLIGHT NUMBER"
        print(FORMAT_INDENT.format(s), end='')
        for flight_num in flight_nums:
            print(str(flight_num)[n], end=' ')
        print()

print_flight_nums()
check_path(EXPECTED_PATHS, Path('.'))
