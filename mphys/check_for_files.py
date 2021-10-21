#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr


# In[4]:


flights = range(330, 355)
instruments = "cdp ds ffssp hvps".split()


# In[3]:


flight_number = 334
instrument = "cdp"
ds = xr.open_dataset(f"current/{instrument}/nc_output/to{flight_number}_{instrument}_r1.nc")


# In[8]:


print("\t".join(["flight_number"] + instruments))
for flight_number in flights:
    p = [str(flight_number)]
    for instrument in instruments:
        try:
            xr.open_dataset(f"current/{instrument}/nc_output/to{flight_number}_{instrument}_r1.nc")
            p.append("1hz file exists")
        except:
            p.append("no file")
    print("\t".join(p))


# In[ ]:




