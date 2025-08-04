# Masters-Thesis

## Overview
This folder contains the scripts written to analyse single-frequency EM data to obtain sea ice thickness measurements in Antarctica. Signal measurements are fit to the One-Dimensional Frequency-domain Electromagnetic Model (ODFEM) created by Irvin (2019). 

The data used span landfast sea ice in McMurdo Sound across four freeze-in dates: August, May, March (2024) and multi-year ice. These areas are separated into two groupds, 'August' and 'Earlier' for analysis. 
- Data collected on Earlier ice was analysed under the assumption that the bulk conductivity of the sub-ice platelet layer was 800 mS/m. Measurements of consolidated ice (fast ice + snow) and sub-ice platelet layer were attained.
- One of the signal response measurements (inphase) was oversaturated on August ice. Therefore, apparent ice thickness measurements were attained on this section - a measurement of consolidated ice thickness + some proportion (30 - 40 %) of the sub-ice platelet layer

A description of each file in this folder is given below. 

Please contact elizabeth.skelton@pg.canterbury.ac.nz with any questions regarding these scripts or analysis. 



IandQResults.py
Plotting raw inphase and quadrature signal measurements 

Plottingpointaverages.py
Simple script to plot the mean average thickness values from point measurements. 

Forwardmodelcurves.py
For making Earlier Ice forward model curves, fitting point measurements and scaling these data to find the appropriate linear regression equations for your dataset.  

EarlierIceAnalysis_ppmtom.py
All part of Earlier Ice analysis: plotting original and scaled inphase and quadrature values on forward model curves. Then inverse distance weighting to obtain CI and SIPL thickness results, then plotting results these thickness results. 

Find_closest.py
Finding EM measurements at drill-hole locations

Radius_allvalues.py
Gathering all EM measurements within a 2m radius of drill hole and calculating their average. 

Filtering_EarlierIce.py 
Isolating EM measurements in an excel sheet to only include those taken on Earlier ice 

Filtering_AugustIce.py 
Isolating EM measurements in an excel sheet to only include those taken on August ice 

StatisticalAnalysis.py 
Analysing the fit of the forward model using point measurement and linear regression for Earlier Ice measurements. 

ApparentThick_Aug.py
Plotting point measurements from August Ice (CI drill-hole vs quadrature) over no-SIPL forward model curves for both HCP and VCP ODFEM coil orientation settings. Then, fitting quadrature EM measurements to the HCP forward model curve to obtain AIT results and save these to excel sheets. It also includes the creation of inphase no-SIPL curves for scaling factor analysis. 

ResultsPlots_AIT.py 
Plotting apparent ice thickness results and overlaying point measurements, for one plot. 

Results_AIT_3plots.py 
Plotting both a single plot and three combined to make one figure. 

ScalingFactors.py
Used to obtain scaling factors from Earlier Ice point measurements following methods outlined by Haas (2021).

ForwardModelCurves_Haas.py 
For creating the forward model curves for inphase and quadrature based on the method outlined by Haas (2021) which assumes a constant CI thickness. 

Combine.py 
For combining multiple excel sheets with the same column headings – used to create entire Earlier Ice and August Ice results spreadsheets. 

Cuttingdata.py
For separating datasets based on coordinates. 

OutlierRemoval_AIT.py
Removing outliers from AIT results. 

PPMtoMeters_AugustIce.py
Used to test forward model curves based on SIPL cond 1000 mS/m for August ice. Not used in the end due to oversaturation of the inphase signal. 

<img width="451" height="685" alt="image" src="https://github.com/user-attachments/assets/587bc74a-0853-467d-9e61-c5199b1a8bdf" />



