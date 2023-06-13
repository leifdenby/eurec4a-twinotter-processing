# coding: utf-8

import xarray as xr


def main():
    ds_1hz = xr.open_dataset("flight339/MASIN/EUREC4A_TO-339_MASIN-1Hz_20200202_v0.6.nc")
    ds_50hz = xr.open_dataset("flight339/MASIN/EUREC4A_TO-339_MASIN-50Hz_20200202_v0.6.nc")

    def _get_longname(v):
        try:
            long_name = ds_1hz[name].long_name
        except KeyError:
            long_name = ds_50hz[name].long_name
        return long_name

    def sortfunc(v):
        if "OXTS" in v:
            return "OXTS"
        if "BAT" in v:
            return "BAT"
        if "LICOR" in v or "LIC" in v or "LICOR" in _get_longname(v):
            return "LICOR"
        return v

    names_50hz = set(filter(lambda v: not v.endswith("_FLAG"), ds_50hz.data_vars))
    names_1hz = set(filter(lambda v: not v.endswith("_FLAG"), ds_1hz.data_vars))
    names = names_1hz.union(names_50hz)

    names = sorted(names, key=sortfunc)
    for name in names:
        p = [name]
        p.append("yes" if name in names_1hz else "no")
        p.append("yes" if name in names_50hz else "no")
        try:
            long_name = ds_1hz[name].long_name
        except KeyError:
            long_name = ds_50hz[name].long_name
        p.append(long_name)
        print("\t".join(p))
