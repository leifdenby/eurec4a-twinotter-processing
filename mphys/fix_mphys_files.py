#!/usr/bin/env python
# coding: utf-8
"""
Utilities for fixing Twin-Otter microphysics data files from the EUREC4A
campaign, specifically:

- make files CF-complianat
- correct timing issue between MASIN data and microphysics data. This is done
  by computing the auto-correlation between the MASIN vertical velocity (not
  the instrument, but environment vertical velocity) and the total particle
  count for a given microphysics instrument. The auto correlation is computed
  for a window of the flight spanning around the time of the maximum vertial
  velocity in the MASIN data. The a period of time before take-off is excluded
  when looking for the maximum vertical velocity (by default 1hr)


Leif Denby, University of Leeds, 2021
"""

import datetime
import textwrap
import warnings
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml
from cfchecker.cfchecks import CFChecker, CFVersion, getargs
from tqdm import tqdm


FLIGHT_NUMBERS = range(330, 355)
MASIN_VAR = "W_OXTS"
MPHYS_VAR = "ambient_particle_number_per_channel"
# window over which to compute correlation and create plots
DT_WINDOW = np.timedelta64(5, "m")
# +- time-span of zoomed window plot
DT_WINDOW_ZOOM = np.timedelta64(30, "s")
# time after take-off to exclude when finding maximum vertical velocity for MASIN data
DT_FROM_TAKEOFF = np.timedelta64(60, "m")
ALL_INSTRUMENTS = ["cdp", "ffssp", "ds", "hvps"]

DATA_ROOT = Path(".")

PLOTS_PATH_ROOT = DATA_ROOT / "plots-fixing"
PLOTS_PATH_ROOT.mkdir(exist_ok=True, parents=True)


def make_mphys_path(source_date, flight_number, instrument, kind):
    return Path(
        f"{kind}/{source_date}/{instrument}/nc_output/to{flight_number}_{instrument}_r1.nc"
    )


def _fix_flag_var(ds, var_name):
    flag_values = ds[var_name].flag_values
    if isinstance(flag_values, str):
        dtype = ds[var_name].dtype
        vals = [
            np.array(s.replace("b", ""), dtype=dtype)
            for s in ds[var_name].flag_values.split(",")
        ]
        flag_values = np.array(vals)
    ds[var_name].attrs["flag_values"] = flag_values
    assert ds[var_name].attrs["flag_values"].dtype == ds[var_name].dtype

    ds[var_name].attrs["flag_meanings"] = format(
        " ".join(ds[var_name].flag_meanings.split())
    )

    return ds[var_name]


def cffix(ds):
    _fix_flag_var(ds=ds, var_name="qc_flag_ambient_particle_number_per_channel")

    # remove coordinates that don't have valid values anyway (they are all nans)
    for v in "longitude latitude".split():
        if v in ds:
            ds = ds.drop_dims(v)

    # drop variables we don't know the value for (also all nans)
    for (
        v
    ) in "altitude platform_speed_wrt_air platform_pitch_angle platform_yaw_angle".split():
        if v in ds:
            ds = ds.drop(v)

    # ensure "coordinates" attributes is removed from encoding so
    # it isn't saved with stored file
    for v in ds.data_vars:
        if "coordinates" in ds[v].encoding:
            del ds[v].encoding["coordinates"]

    # CF-conventions stipulates that "Conventions" attribute should be with capital C
    ds.attrs["Conventions"] = ds.attrs.pop("conventions")

    return ds


class ValidationError(Exception):
    pass


class MyCFChecker(CFChecker):
    _logged_messages = []

    def _add_message(self, category, msg, var=None, code=None):
        self._logged_messages.append(locals())


def _check_file(filename, show_warnings=True):
    arglist = ["", filename]

    (
        badc,
        coards,
        debug,
        uploader,
        useFileName,
        regionnames,
        standardName,
        areaTypes,
        cacheDir,
        cacheTables,
        cacheTime,
        version,
        files,
    ) = getargs(arglist)

    # this will attempt auto-finding of version
    version = CFVersion()

    inst = MyCFChecker(
        uploader=uploader,
        useFileName=useFileName,
        badc=badc,
        coards=coards,
        cfRegionNamesXML=regionnames,
        cfStandardNamesXML=standardName,
        cfAreaTypesXML=areaTypes,
        cacheDir=cacheDir,
        cacheTables=cacheTables,
        cacheTime=cacheTime,
        version=version,
        debug=False,
        silent=True,
    )

    inst.checker(filename)

    file_errors = {}
    file_warnings = {}

    for lm in inst._logged_messages:
        category = lm["category"]
        msg = textwrap.indent(textwrap.fill(lm["msg"]), prefix="    ")
        var = lm["var"] if lm["var"] else "__global__"
        if category == "WARN":
            var_warnings = file_warnings.setdefault(var, [])
            var_warnings.append(msg)
        elif category == "ERROR":
            var_errors = file_errors.setdefault(var, [])
            var_errors.append(msg)
        elif category in ["VERSION", "INFO"]:
            pass
        else:
            raise NotImplementedError(category)

    if len(file_errors) > 0:
        print("The following errors were detected:")
        for var in sorted(file_errors.keys()):
            print(f"  {var}:")
            for error in file_errors[var]:
                print(f"{error}")
    else:
        if show_warnings:
            print("no errors!")

    if show_warnings and len(file_warnings) > 0:
        print()
        print("The following warnings were raised:")
        for var in sorted(file_warnings.keys()):
            print(f"  {var}:")
            for warning in file_warnings[var]:
                print(f"{warning}")


class EmptyFileException(Exception):
    pass


def load_cffixed_mphys_ds(flight_number, source_date, instrument):
    fn_src = make_mphys_path(
        source_date=source_date,
        flight_number=flight_number,
        instrument=instrument,
        kind="source",
    )
    fn_cffixed = make_mphys_path(
        source_date=source_date,
        flight_number=flight_number,
        instrument=instrument,
        kind="cf-fixed",
    )

    if not fn_cffixed.exists():
        ds = xr.open_dataset(fn_src)
        if not "qc_flag_ambient_particle_number_per_channel" in ds:
            # this file is empty...
            raise EmptyFileException("Empty file")
        ds = ds.copy()
        ds = cffix(ds)
        fn_cffixed.parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(fn_cffixed)

        # it appears the filesystem on JASMIN is sometimes a bit slow, so let's
        # wait until the file appears
        while True:
            if fn_cffixed.exists():
                break
            sleep(0.5)

        _check_file(str(fn_cffixed), show_warnings=False)

    return xr.open_dataset(fn_cffixed)


def load_masin_ds(flight_number):
    fn_masin = f"/home/users/lcdenby/eurec4auk/public/data/obs/MASIN/EUREC4A_TO-{flight_number}_MASIN-1Hz_*_v0.6.nc"
    return xr.open_mfdataset(fn_masin).rename(dict(Time="time"))


def extract_masin_window(ds_masin, dt_window, dt_fromtakeoff):
    ds_ = ds_masin.sel(time=slice(ds_masin.time.min() + dt_fromtakeoff, None))
    tn_w_max = ds_.W_OXTS.argmax(dim="time").compute().data
    t_w_max = ds_.isel(time=tn_w_max).time

    return ds_masin.sel(time=slice(t_w_max - dt_window, t_w_max + dt_window))


def calc_offset(da1, da2, show_plot=False):
    v1 = da1.fillna(0.0).values
    v2 = da2.fillna(0.0).values
    vv = np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2).conjugate()).real

    N = vv.shape[0]
    da_vv = xr.DataArray(
        np.roll(vv, N // 2), dims=("offset"), coords=dict(offset=np.arange(N) - N // 2)
    )
    offset_max_corr = int(da_vv.isel(offset=da_vv.argmax(dim="offset")).offset.data)

    if show_plot:
        Ns = 1000
        fig, ax = plt.subplots()
        da_vv.sel(offset=slice(-Ns, Ns)).plot(ax=ax)
        ax.axvline(offset_max_corr, color="red")

        return (fig, ax), offset_max_corr
    else:
        return offset_max_corr


def plot_comparison(da1, da2):
    fig, axes = plt.subplots(nrows=2, figsize=(14, 6), sharex=True)

    ax = axes[0]
    da1.plot(ax=ax)

    ax = axes[1]
    da2_window = da2.sel(time=slice(da1.time.min(), da1.time.max()))
    da2_window.plot(ax=ax, color="red")

    da2_window.plot(ax=axes[0].twinx(), alpha=0.3, color="red")

    return fig, axes


def plot_comparison_all(da_masin, **das_mphys):
    n_instruments = len(das_mphys) + 1
    fig, axes = plt.subplots(
        nrows=n_instruments, figsize=(14, 3 * n_instruments), sharex=True
    )

    ax = axes[0]
    da_masin.plot(ax=ax)
    instruments_mphys = das_mphys.keys()
    ax.set_title(f"MASIN with {', '.join(instruments_mphys)}")

    colors = sns.color_palette(n_colors=n_instruments + 1)[1:]

    for ax, instrument, da_mphys, color in zip(
        axes[1:], das_mphys.keys(), das_mphys.values(), colors
    ):
        da_mphys_window = da_mphys.sel(
            time=slice(da_masin.time.min(), da_masin.time.max())
        )
        da_mphys_window.plot(ax=ax, color=color)
        ax.set_title(instrument)

        ax_twin = axes[0].twinx()
        ax_twin.axis("off")
        da_mphys_window.plot(ax=ax_twin, alpha=0.3, color=color)

    fig.tight_layout()

    for ax in axes:
        ax.grid(visible=True, which="major")
        ax.grid(visible=True, which="minor", alpha=0.2)
        ax.minorticks_on()

    return fig, axes


def getset_offset(flight_number, instrument, fn_offsets, new_value=None):
    "get/set the offset value by maintaining a csv-file as database of offsets"
    if not Path(fn_offsets).exists():
        if new_value:
            data = {flight_number: {instrument: new_value}}
        else:
            data = {flight_number: {}}
        df = pd.DataFrame(data.values(), index=data.keys())
        df.index.name = "flight_number"
    else:
        df = pd.read_csv(fn_offsets, comment="#", index_col="flight_number")
        if new_value:
            df.loc[flight_number, instrument] = new_value

    if new_value is not None:
        with open(fn_offsets, "w") as fh:
            fh.write(
                "# Timing offsets (in seconds) for microphysics instruments computed\n"
            )
            fh.write("# using auto-correlation with MASIN vertical velocity\n")
            fh.write("# for observations made during the 2020 EUREC4A campaign\n")
            fh.write(
                f"# {datetime.datetime.now().isoformat()}, Leif Denby, University of Leeds\n\n"
            )
            df.to_csv(fh)

    try:
        val = df.loc[flight_number, instrument]
        if np.isnan(val):
            return None
        return int(val)
    except KeyError:
        return None


def extract_over_window(
    ds_mphys, mphys_var, ds_masin, masin_var, dt_window, dt_fromtakeoff
):
    # first we need to ensure that the MASIN data we examine for
    # finding the max vertical velocity actually is within the time
    # range where we have mphys data
    da_mphys_times_with_data = ds_mphys.where(
        ~ds_mphys[mphys_var].isnull(), drop=True
    ).time
    da_mphys_data_trange = (
        da_mphys_times_with_data.min(),
        da_mphys_times_with_data.max(),
    )
    ds_masin_with_mphys = ds_masin.sel(time=slice(*da_mphys_data_trange))

    ds_masin_window = extract_masin_window(
        ds_masin=ds_masin_with_mphys, dt_window=dt_window, dt_fromtakeoff=dt_fromtakeoff
    )
    da_masin_window = ds_masin_window[masin_var]

    ds_mphys_window = ds_mphys.sel(
        time=slice(ds_masin_window.time.min(), ds_masin_window.time.max())
    )

    da_mphys_window = ds_mphys_window[mphys_var].sum(dim="index")

    return da_masin_window, da_mphys_window


def _make_comparison_plots(
    kind, flight_number, da_masin_window, plots_path, **das_mphys_window
):
    """
    kind: [original, offset]
    """
    if len(das_mphys_window) == 1:
        instrument = next(iter(das_mphys_window.keys()))
    else:
        instrument = "all"
    plot_id = f"TO{flight_number}.MASIN_vs_{instrument}"
    fn_comparison = f"{plot_id}.{kind}.png"
    fn_comparison_zoom = f"{plot_id}.{kind}.zoom.png"

    plots_path.mkdir(exist_ok=True, parents=True)

    avail_instruments = das_mphys_window.keys()
    fig, axes = plot_comparison_all(da_masin_window, **das_mphys_window)
    fig.suptitle(
        f"Flight number {flight_number}, original data. MASIN, {', '.join(avail_instruments)}",
        y=1.05,
    )
    fig.savefig(PLOTS_PATH_ROOT / fn_comparison, bbox_inches="tight")
    t_center = da_masin_window.time.min() + DT_WINDOW
    for ax in axes:
        ax.set_xlim(t_center - DT_WINDOW_ZOOM, t_center + DT_WINDOW_ZOOM)
    fig.savefig(plots_path / fn_comparison_zoom, bbox_inches="tight")
    plt.close(fig)
    del fig


def _calc_offsets_for_flight_instrument(
    ds_masin, flight_number, instrument, fn_offsets
):
    ds_mphys = load_cffixed_mphys_ds(flight_number=flight_number, instrument=instrument)
    if not MPHYS_VAR in ds_mphys:
        # some mphys files don't contain any data...
        return None

    da_masin_window, da_mphys_window = extract_over_window(
        ds_mphys=ds_mphys,
        mphys_var=MPHYS_VAR,
        masin_var=MASIN_VAR,
        ds_masin=ds_masin,
        dt_window=DT_WINDOW,
        dt_fromtakeoff=DT_FROM_TAKEOFF,
    )

    plot_id = f"TO{flight_number}.MASIN_vs_{instrument}"
    fn_correlation = f"{plot_id}.{MASIN_VAR}_npart_correll.png"

    _make_comparison_plots(
        kind="original",
        flight_number=flight_number,
        da_masin_window=da_masin_window,
        plots_path=PLOTS_PATH_ROOT / "calc",
        **{instrument: da_mphys_window},
    )

    (fig, ax), offset = calc_offset(
        da1=da_masin_window,
        da2=da_mphys_window.interp_like(da_masin_window),
        show_plot=True,
    )
    fig.savefig(PLOTS_PATH_ROOT / fn_correlation)
    plt.close(fig)
    del fig

    ds_mphys_offset = ds_mphys.roll(time=offset, roll_coords=False)
    ds_mphys_offset_window = ds_mphys_offset.sel(
        time=slice(da_masin_window.time.min(), da_masin_window.time.max())
    )
    da_mphys_offset_window = ds_mphys_offset_window[MPHYS_VAR].sum(dim="index")

    _make_comparison_plots(
        kind="offset",
        flight_number=flight_number,
        da_masin_window=da_masin_window,
        plots_path=PLOTS_PATH_ROOT / "calc",
        **{instrument: da_mphys_offset_window},
    )

    getset_offset(
        flight_number=flight_number,
        instrument=instrument,
        new_value=offset,
        fn_offsets=fn_offsets,
    )


def calc_offsets(source_date, fn_offsets, only__compute_missing=True):
    instruments = ALL_INSTRUMENTS
    for flight_number in tqdm(FLIGHT_NUMBERS, desc="flight"):
        ds_masin = load_masin_ds(flight_number=flight_number)
        ds_masin.attrs["flight_number"] = flight_number
        for instrument in tqdm(instruments, desc="inst", leave=False):
            try:
                offset = getset_offset(
                    flight_number=flight_number,
                    instrument=instrument,
                    fn_offsets=fn_offsets,
                )
                if only__compute_missing and offset is not None:
                    continue

                _calc_offsets_for_flight_instrument(
                    ds_masin=ds_masin,
                    flight_number=flight_number,
                    instrument=instrument,
                    fn_offsets=fn_offsets,
                )
            except (FileNotFoundError, EmptyFileException) as ex:
                continue
            except Exception as ex:
                print(f"There was an issue with TO{flight_number} {instrument}")
                raise


def process_with_selected_offsets(
    fn_offsets, offset_source, instruments, source_date, save_processed_files=False
):

    # make plots with the best offset
    for flight_number in tqdm(FLIGHT_NUMBERS, desc="flight"):
        ds_masin = load_masin_ds(flight_number=flight_number)
        offset = getset_offset(
            flight_number=flight_number, instrument=offset_source, fn_offsets=fn_offsets
        )
        if offset in [None, "exclude"]:
            print(f"skipping {flight_number}, offset: {offset}")
            continue

        for instrument in tqdm(instruments, desc="inst", leave=False):
            datasets = {}
            for instrument in instruments:
                try:
                    datasets[instrument] = load_cffixed_mphys_ds(
                        flight_number=flight_number,
                        instrument=instrument,
                        source_date=source_date,
                    )
                except (FileNotFoundError, EmptyFileException):
                    continue

            da_masin_window, _ = extract_over_window(
                ds_mphys=list(datasets.values())[0],
                mphys_var=MPHYS_VAR,
                masin_var=MASIN_VAR,
                ds_masin=ds_masin,
                dt_window=DT_WINDOW,
                dt_fromtakeoff=DT_FROM_TAKEOFF,
            )

            das_mphys_window = {}
            das_mphys_offset_window = {}
            for instrument, ds_mphys in datasets.items():

                t_range_window = (
                    da_masin_window.time.min(),
                    da_masin_window.time.max(),
                )
                da_mphys_window = (
                    ds_mphys[MPHYS_VAR]
                    .sel(time=slice(*t_range_window))
                    .sum(dim="index")
                )
                das_mphys_window[instrument] = da_mphys_window

                ds_mphys_offset = ds_mphys.roll(time=offset, roll_coords=False)
                ds_mphys_offset_window = ds_mphys_offset.sel(
                    time=slice(da_masin_window.time.min(), da_masin_window.time.max())
                )
                da_mphys_offset_window = ds_mphys_offset_window[MPHYS_VAR].sum(
                    dim="index"
                )
                das_mphys_offset_window[instrument] = da_mphys_offset_window

            _make_comparison_plots(
                flight_number=flight_number,
                kind="original",
                da_masin_window=da_masin_window,
                plots_path=PLOTS_PATH_ROOT / "processed" / offset_source,
                **das_mphys_window,
            )

            _make_comparison_plots(
                flight_number=flight_number,
                kind="offset",
                da_masin_window=da_masin_window,
                plots_path=PLOTS_PATH_ROOT / "processed" / offset_source,
                **das_mphys_offset_window,
            )

            for instrument, ds_mphys in datasets.items():
                fn_processed = make_mphys_path(
                    source_date=source_date,
                    flight_number=flight_number,
                    instrument=instrument,
                    kind="processed",
                )
                fn_processed.parent.mkdir(exist_ok=True, parents=True)
                ds_mphys.to_netcdf(fn_processed)


def optional_debugging(with_debugger):
    """
    Optionally catch exceptions and launch ipdb
    """
    if with_debugger:
        import ipdb

        return ipdb.launch_ipdb_on_exception()
    else:

        class NoDebug:
            def __enter__(self):
                pass

            def __exit__(self, *args, **kwargs):
                pass

        return NoDebug()


def main():
    only__compute_missing = True
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("source_date")
    argparser.add_argument("offset_fn")
    argparser.add_argument("--calc", default=False, action="store_true")
    argparser.add_argument("--process-with", default=None)
    argparser.add_argument("--debug", default=False, action="store_true")

    args = argparser.parse_args()

    with optional_debugging(with_debugger=args.debug):
        if args.calc:
            calc_offsets(source_date=args.source_date, fn_offsets=args.offset_fn)

        if args.process_with:
            process_with_selected_offsets(
                source_date=args.source_date,
                fn_offsets=args.offset_fn,
                offset_source=args.process_with,
                instruments=ALL_INSTRUMENTS,
            )

    if not args.calc and not args.process_with:
        print("please set one of --calc or --process-with")
        argparser.help()


if __name__ == "__main__":
    main()
