# Copies all datafiles of a specific revision to `public_html` so that they are
# accessible via HTTP

if [ -z "$1" ]; then
  echo "$0 <revision_num>"
  exit 1
fi

PUB_PATH="/gws/nopw/j04/eurec4auk/public/data/obs"
CWD=`pwd -P`

# Make sure globstar is enabled
shopt -s globstar

# MASIN obs data
#for i in **/MASIN*${1}.nc; do # Whitespace-safe and recursive
  #echo "link $i"
#done

echo "Looking for dataset in ${CWD}"

# remove symlinks for current public release
find $PUB_PATH/MASIN/ -wholename "*MASIN*_${1}.nc" -exec rm {} \;
# create symlinks for new release
find flight*/MASIN/ -wholename "*MASIN*_${1}.nc" -exec ln -s $CWD/{} $PUB_PATH/MASIN \;
# print content of public folder
echo "public datasets (in ${PUB_PATH}):"
find $PUB_PATH/MASIN/ -wholename "*MASIN*_${1}.nc" -exec echo {} \;

cp masin_processing/CHANGELOG.txt $PUB_PATH/MASIN



# todo: make symlinks to revision files and changelog and copy to public dir
