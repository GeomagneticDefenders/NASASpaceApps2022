# NASASpaceApps2022

Our challenge was to design an algorithm to improve solar wind data quality from the satellite DSCVR in order to make better predictions of solar activity events like solar flares, coronal mass ejections and solar storms. We used a variety of techniques and ultimately designed a neural network to predict wind speed, density and temperature from the magnetic field data provided. More accurate predictions and data collection allows for better mitigation of potential solar hazards. In fact, we developed a series of infographics with the different categories of solar events with information on what to expect and a small aurora borealis predictor based on solar wind data.

test.py contains the code to plot the z component of Wind's and DSCVR's magnetic field for a given day.
carrington.py loads the data for January 2022 and creates a csv file where each row contains the three
components of the vector and the corresponding wind speed, temeperature and density.
nnet.py is our small scale SVM model to predict the wind parameters from the magnetic field dataset
