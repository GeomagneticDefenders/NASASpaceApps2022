from spacepy import pycdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import csv

bx = []
dtime = []

v = []
dens = []
temp = []
wtime = []

# Remove incorrect measurements
def check_vec(vec):
    t = []
    for comp in vec:
        t.append(abs(comp) < 9999)
    return all(t)

# Parse Magnetic Field Vectors and wind speed, temperature and density
for n, (dfile, wfile) in enumerate(zip(os.listdir("dscvr/"), os.listdir('ions/'))):
    dscvr = pycdf.CDF("dscvr/" + dfile)
    dvecs = [[*i] if check_vec(i) else np.nan for i in dscvr['B1GSE'][::10]]
    dti = dscvr['Epoch1'][::10]
    bx.extend(dvecs)
    dtime.extend(dti)

    wind = pycdf.CDF("ions/" + wfile)
    wvel = [-i if not np.isclose(i, 0) and i < 10000 else np.nan for i in wind['Proton_VX_nonlin']]
    wt = wind['Epoch']
    v.extend(wvel)
    dens.extend([i if not np.isclose(i, 0) and i < 10000 else np.nan for i in wind['Proton_Np_nonlin']])
    temp.extend([i if not np.isclose(i, 0) and i < 9999 else np.nan for i in wind['Proton_W_nonlin']])
    wtime.extend(wt)

print("done")

matches = 0

m = {}

filtered_data = []

for n, small_t in enumerate(wtime):
    print(n)
    for j, big_t in enumerate(dtime[n*10:n*22]):
        if abs(big_t.timestamp() - small_t.timestamp()) < 10 and j not in m.values():
            if isinstance(bx[j], list):
                filtered_data.append([*bx[j], v[n], dens[n], temp[n]])
                m[n] = j
                matches += 1
                # print(filtered_data[-1])
    if n >= 5000:
        break

print("Matches", matches)
    
header = ['bx', 'by', 'bz', 'v', 'n', 'w']

with open('data2.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for r in filtered_data:
        writer.writerow(r)

# print(len(bx))

# fig, ax = plt.subplots()
# # ax.plot(dtime, bx, linewidth=1.0)
# ax.plot(wtime, temp, linewidth=1.0)

# plt.show()