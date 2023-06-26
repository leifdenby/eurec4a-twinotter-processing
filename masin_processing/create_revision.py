#!/usr/bin/env python
from pathlib import Path
import xarray as xr
import numpy as np
import parse
from datetime import datetime
import cfunits

import shutil

DATAFILE_FORMAT = {
    'MASIN': "core_masin_{date}_r{rev}_flight{flight_num}_{freq}hz.nc"
}

DATE_FORMAT = {
    'MASIN': '%Y%m%d',
    'EUREC4A': '%Y%m%d',  # note: there is ambiguity about this right now, might need to change to month before day
}

DATAFILE_PATH = "flight{flight_num}/{instrument}"
# see https://github.com/eurec4a/meta/blob/master/naming_conventions.md
# <campaign_id>_<project_id>_<platform_id>_<instrument_id>_<variable_id>_<time_id>_<version_id>.nc
EUREC4A_FILE_FORMAT = "EUREC4A_{platform_id}_{instrument_id}_{time_id}_{version_id}.nc"
EUREC4A_REF_TIME = datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)

ALL_FLIGHT_NUMBERS = range(330, 355)

def find_most_recent_revision_source_file(data_root, flight_num, instrument, freq):
    if instrument in DATAFILE_FORMAT:
        # ignore the revision info in the file name
        instrument_path = DATAFILE_PATH.format(flight_num=flight_num, instrument=instrument)
        fn_pattern = DATAFILE_FORMAT[instrument].format(flight_num=flight_num, date="*", rev="*", freq=freq)
        fullpath = data_root / instrument_path
        
        filepaths = list(fullpath.glob(fn_pattern))
        if len(filepaths) == 0:
            raise Exception(f"Didn't find any files for flight {flight_num}")
        elif len(filepaths) > 1:
            # get file with highest revision number
            def _get_rev(fp):
                return parse.parse(DATAFILE_FORMAT[instrument], fp.name)["rev"]
            filepaths = list(sorted(filepaths, key=_get_rev))
            filepath = filepaths[-1]
        else:
            filepath = filepaths[0]
        return filepath
    else:
        raise NotImplementedError(instrument)
    
def find_most_recent_processed_version(data_root, flight_num, instrument, freq):
    time_id = "*"
    platform_id = f"TO-{flight_num}"
    instrument_id = f"{instrument}-{freq}Hz"
    fn_new = EUREC4A_FILE_FORMAT.format(
        platform_id=platform_id,
        instrument_id=instrument_id,
        time_id=time_id,
        version_id="*",
    )
    flight_datapath = Path(data_root) / DATAFILE_PATH.format(instrument=instrument, flight_num=flight_num)
    print(f"Looking in {flight_datapath} for {fn_new}")
    filepaths = list(flight_datapath.glob(fn_new))

    if len(filepaths) == 0:
        raise Exception(f"Didn't find any files for flight {flight_num}")
    elif len(filepaths) > 1:
        # get file with highest version number
        def _get_version(fp):
            return parse.parse(EUREC4A_FILE_FORMAT, fp.name)["version_id"]
        filepaths = list(sorted(filepaths, key=_get_version))
        filepath = filepaths[-1]
    else:
        filepath = filepaths[0]
    return filepath

def _correct_radar_alt_and_create_composite_alt(ds):
    """
    Correct radar altitude measurements (HGT_RADR1) which were previously
    scaled by -2.0 and add composite variable using radar and gps altitude to
    produce height above ocean (ALT_COMPOSITE)
    """
    z_threshold = 500

    if "corrected" not in ds.HGT_RADR1.attrs.get("notes", ""):
        # correct radar altitude
        attrs = ds.HGT_RADR1.attrs
        ds["HGT_RADR1"] = -2.0 * ds.HGT_RADR1
        ds.HGT_RADR1.attrs.update(attrs)
        ds.HGT_RADR1.attrs["notes"] = "corrected -2.0 scaling error from instrument output"

    # find points where TO was flying low-level east of Barbados
    ds_lowlevel_ocean = (
        ds.where(ds.HGT_RADR1 > 50.0)
        .where(ds.HGT_RADR1 < z_threshold)
        .where(ds.LON_OXTS > -59.4)
    )

    # compute mean altitude offset between the radar altitude and GPS altitude for these points
    da_ocean_offset = (ds_lowlevel_ocean.ALT_OXTS - ds_lowlevel_ocean.HGT_RADR1).mean(
        skipna=True
    )

    # create a altitude composite using the radar altitude when below 400m
    # otherwise use the offset GPS altitude
    ds["ALT_OXTS_OFFSET"] = ds.ALT_OXTS - da_ocean_offset
    ds.ALT_OXTS_OFFSET.attrs["units"] = "m"
    ds.ALT_OXTS_OFFSET.attrs[
        "long_name"
    ] = f"{ds.ALT_OXTS.long_name} offset with radar below {z_threshold}m altitude to measure ocean-relative height"

    ds["ALT_COMPOSITE"] = ds.HGT_RADR1.where(ds.HGT_RADR1 < z_threshold, ds.ALT_OXTS_OFFSET)
    ds.ALT_COMPOSITE.attrs["units"] = "m"
    ds.ALT_COMPOSITE.attrs[
        "long_name"
    ] = "altitude relative to ocean surface (radar + gps composite)"


def main(source_dir, version, changelog, dry_run):
    instrument = "MASIN"
    freq = "1"
    t_now = datetime.now()
    
    for flight_num in ALL_FLIGHT_NUMBERS:
        filepath = find_most_recent_revision_source_file(data_root=source_dir, instrument=instrument, flight_num=flight_num, freq=freq)


        print(f"{filepath.name}:")
        r = parse.parse(DATAFILE_FORMAT[instrument], filepath.name)

        ds = xr.open_dataset(filepath, decode_times=False)

        nc_rev = ds.attrs['Revision']
        filename_rev = r['rev']
        version_id = f"v{version}"
        print(f"  exising revision info")
        print(f"    filename: {filename_rev}")
        print(f"    nc-attrs: {nc_rev}")
        print(f"  new version: {version_id}")

        if instrument == 'MASIN':
            time_units = ds.Time.attrs['units']
            # ensure that time-spacing is constant
            dt_all = np.diff(ds.Time.values)
            assert dt_all.max() == dt_all.min()
            # correct o follow EUREC4A time reference
            if "milliseconds" in time_units:
                # do nothing, can't count milliseconds as 32bit ints from
                # 2020-01-01 because we overflow on 2020-01-25
                pass
            else:
                nc_tref = cfunits.Units(time_units).reftime
                t_offset = EUREC4A_REF_TIME - nc_tref
                if dry_run:
                    print(f"  would adjust reference time to 2020-01-01 00:00:00, by {t_offset} ({t_offset.total_seconds()}s)")
                    print(f"  current time units: {time_units}")
                else:
                    ds.Time.values -= int(t_offset.total_seconds())
                    ds.Time.attrs['units'] = EUREC4A_REF_TIME.strftime(
                        'seconds since %Y-%m-%d %H:%M:%S +0000 UTC'
                    )
                    
            if dry_run:
                print("  would correct HGT_RADR1 and add composite altitude ALT_COMPOSITE")
            else:
                _correct_radar_alt_and_create_composite_alt(ds=ds)

            old_rev_info = f"filename: `{filename_rev}`, nc-attrs: `{nc_rev}`"
            history_s = f"version created by Leif Denby {t_now.isoformat()}, existing revision info: {old_rev_info} from file: `{filepath.name}`"
            if dry_run:
                print(f"  would set ds.attrs['version'] = {version_id}")
                print(f"  would set ds.attrs['history'] = {history_s}")
                print(f"  would delete ds.attrs['Revision']")
            else:
                ds.attrs['version'] = version_id
                ds.attrs['history'] = history_s
                ds.attrs['contact'] = "Tom Lachlan-Cope <tlc@bas.ac.uk>, Leif Denby <l.c.denby@leeds.ac.uk>"
                ds.attrs['acknowledgement'] = "TO NOT USE FOR PUBLICATION! EARLY-RELEASE DATA"
                del ds.attrs['Revision']

            # fixes for masin data
            for v in ds.data_vars:
                if not "units" in ds[v].attrs:
                    if v in ["Time"]:
                        pass
                    else:
                        raise Exception(v, ds[v])
                elif type(ds[v].attrs['units']) == np.int8 and ds[v].attrs['units'] == 1:
                    # should be string, not a number
                    ds[v].attrs['units'] = "1"
            # make time the main coordinate
            ds = ds.swap_dims(dict(data_point='Time'))

            ds.attrs['flight_number'] = r['flight_num']

            date_filename = datetime.strptime(r['date'], DATE_FORMAT[instrument])
            time_id = date_filename.strftime(DATE_FORMAT['EUREC4A'])
            platform_id = f"TO-{r['flight_num']}"
            instrument_id = f"{instrument}-{r['freq']}Hz"
            fn_new = EUREC4A_FILE_FORMAT.format(
                platform_id=platform_id,
                instrument_id=instrument_id,
                time_id=time_id,
                version_id=version_id,
            )
            p_out = Path(DATAFILE_PATH.format(instrument=instrument, flight_num=r['flight_num']))/fn_new
            if p_out.exists():
                raise Exception(f"`{p_out}` exists, aborting")
            if dry_run:
                print(f"  would write to {p_out}")
            else:
                ds.to_netcdf(p_out)
        print(flush=True)

    changelog_extra = f"""

# {t_now.date().isoformat()} {version_id}
{changelog}
"""
    if dry_run:
        print(f"would add to CHANGELOG: {changelog_extra}")
    else:
        with open("masin_processing/CHANGELOG.txt", "a") as fh:
            fh.write(changelog_extra)


if __name__ == "__main__":
    def version_str(s):
        vals = s.split(".")
        assert len(vals) == 2
        [int(v) for v in vals]
        return s

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('source_dir', type=Path)
    argparser.add_argument('--version', required=True, type=version_str)
    argparser.add_argument('--changelog', required=True)
    argparser.add_argument('--dry-run', action='store_true', default=False)
    args = argparser.parse_args()

    print(f"Looking for files in {args.source_dir}")

    main(**dict(vars(args)))
