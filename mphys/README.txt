# microphysics data from the Twin-Otter

This directory contains microphysics observations from the Twin-Otter provided
by Gary Lloyd at Manchester as h5 and netcdf files. These files have a number
of issues:

1. The files are no CF-compliant (don't pass the CF-checker)
2. A number of files have one-second and larger gaps (with timesteps missing)
3. There is a timing issue between the microphysics observations and the MASIN
   observations

This script `fix_mphys_files.py` resolves all these issues. The first step is
to compute the offsets using autocorrelation analysis between MASIN vertical
velocity and the total droplet number (saved in a CSV) file

$> python fix_mphys_files.py [source_date] [offsets-csv-filename] --calc

$> python fix_mphys_files.py 20210614 offsets.csv --calc

input:
- `{source}/{source_date}/{instrument}/nc_output/*.nc`

output:
- offsets-csv-filename
    contains a row for each flight and an offset computed for each microphysics
    instrument as the column.

- `plots-fixing/{source_date}/calc`
    plots of the offsetting

- `cf-fixed/{source_date}/{instrument}/nc_output`
    CF-compliant intermediate files

The final corrected files can be created from a specific column in the offsets
CSV by passing in the name of the column with the offsets to use

$> python fix_mphys_files.py [source_date] [offsets-csv-filename] --process-with [offset_source]

$> python fix_mphys_files.py 20210614 offsets.csv --process-with v1

input:
- `{source}/{source_date}/{instrument}/nc_output/*.nc`

output:
- offsets-csv-filename
    contains a row for each flight and an offset computed for each microphysics
    instrument as the column.

- `plots-fixing/{source_date}/{offset_source}`
    plots of the offsetting

- `cf-fixed/{source_date}/{instrument}/nc_output`
    CF-compliant intermediate files

- `processed/{source_date}/{instrument}/nc_output`
    CF-compliant offset files
