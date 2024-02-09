# mass_con
Glacier mass conservation tools

### file overview
- *config.ini* Configuration file for FlowlineMassCon.py
- *GenFlowlines.m* Matlab script for generating x-y coordinate vertices along glacier flowlines
- *FlowlineMassCon.py* Python3 package to solve for ice thickness along glacier flowlines through mass conservation - based on McNabb et al., 2012
- *flowline_wrapper.sh* Bash shell script for FlowlineMassCon.py over a range of input mass balance parameters 
- *FluxGate.ipynb* Jupyter Notebook which solves for ice thickness across flux gates - currently set up for Ruth Glacier inputs