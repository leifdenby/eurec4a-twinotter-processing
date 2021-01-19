# Copies all datafiles of a specific revision to `public_html` so that they are
# accessible via HTTP

if [ -z "$1" ]; then
  echo "$0 <revision_num>"
  exit 1
fi

PUB_PATH="/gws/nopw/j04/eurec4auk/public/data/obs"

# MASIN obs data

# todo: make symlinks to revision files and changelog and copy to public dir
