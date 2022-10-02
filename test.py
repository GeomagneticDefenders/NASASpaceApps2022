from spacepy import pycdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from dtw import *

# This program plots Wind and DSCVR Magentig Field z component

wind = pycdf.CDF(r"wi_h2_mfi_20220101_v04.cdf")
dscvr = pycdf.CDF(r"dscovr_h0_mag_20220101_v01.cdf")

plt.style.use('dark_background')
fig, ax = plt.subplots()

wvecs = [i[2] for i in wind['BGSE'][::100]]
wtime = [i[0] for i in wind['Epoch'][::100]]

dvecs = [i[2] if not np.isclose(i[2], 0) and i[2] > -10000 else np.nan for i in dscvr['B1GSE'][::10]]
dtime = dscvr['Epoch1'][::10]

shifted_time = []
for i in dtime:
    shifted_time.append(i + dt.timedelta(seconds=17))

vecs = [[i,j] for i,j in zip(wvecs, dvecs)]

ax.plot(wtime, wvecs, linewidth=1.0, label='Wind') # ORANGE 
ax.plot(shifted_time, dvecs, linewidth=1.0, label='DSCVR') # BLUE

# fig, ax = plt.subplots()
plt.xlabel("Time")
plt.ylabel("Bz[nT]")

plt.legend()
plt.show()