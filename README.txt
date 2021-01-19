Transfer examples

To only copy MASIN data to local machine:

    $> rsync -am --include='flight*/MASIN/*.nc' --include='*/' --exclude='*' jasmin-xfer1.ceda.ac.uk:/gws/nopw/j04/eurec4auk/data/obs .
