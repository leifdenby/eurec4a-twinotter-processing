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

def _find_source_files(source_dir, flight_num, instrument, freq):
    if instrument in DATAFILE_FORMAT:
        # ignore the revision info in the file name
        instrument_path = DATAFILE_PATH.format(flight_num=flight_num, instrument=instrument)
        fn_pattern = DATAFILE_FORMAT[instrument].format(flight_num=flight_num, date="*", rev="*", freq=freq)
        fullpath = source_dir / instrument_path
        return list(fullpath.glob(fn_pattern))
    else:
        raise NotImplementedError(instrument)


def main(source_dir, version, changelog, dry_run):
    instrument = "MASIN"
    freq = "1"
    t_now = datetime.now()
    
    for flight_num in ALL_FLIGHT_NUMBERS:
        files = _find_source_files(source_dir=source_dir, instrument=instrument, flight_num=flight_num, freq=freq)

        if len(files) == 0:
            raise Exception(f"Didn't find any files for flight {flight_num}")
        elif len(files) > 1:
            # get file with highest revision number
            def _get_rev(fp):
                return parse.parse(DATAFILE_FORMAT[instrument], fp.name)["rev"]
            files = list(sorted(files, key=_get_rev))
            filepath = files[-1]
        else:
            filepath = files[0]

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
                if type(ds[v].attrs['units']) == np.int8 and ds[v].attrs['units'] == 1:
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
        with open("masin-processing/CHANGELOG.txt", "a") as fh:
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
