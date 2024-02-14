# mass_con
Glacier mass conservation tools

### File Overview

- *FluxGate.ipynb* Jupyter Notebook which solves for ice thickness across flux gates - currently set up for Ruth Glacier inputs
- *GenFlowlines.m* Matlab script for generating x-y coordinate vertices along glacier flowlines
- *config.ini* Configuration file for FlowlineMassCon.py
- *flowline_wrapper.sh* Bash shell script for FlowlineMassCon.py over a range of input mass balance parameters 
- *FlowlineMassCon.py* Python3 package to solve for ice thickness along glacier flowlines through mass conservation - based on McNabb et al., 2012
  - This method follows the following schematic:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/btobers/mass_con/blob/main/recs/flowline_masscon_schematic.jpg" height="300"><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gray arrows represent glacier flow vectors, with downstream towards to south-southeast. The glacier surface is discretized into an irregular grid, by connecting vertices along glacier flowlines (thick black lines) and transverse to flowlines (thin black lines). In the blown-up view, cell $c_1$ is composed of vertices (x’s) $v_{1,0}$, $v_{1,1}$, $v_{2,1}$, and $v_{2,0}$, and has surface area $S_1$. The ice influx to this cell is $q_1$, and the ice efflux is $q_2$. The ice efflux from cell $c_1$ is approximated by the product of the average ice thickness across the downstream boundary, the boundary length $l_2$, the surface velocity normal to the downstream boundary $v_{sfc_2}$ (omitted from schematic for simplicity but collocated with vector $q_2$), and $\gamma$, the factor which relates the observed surface velocity to the depth-averaged velocity. Figure adapted from McNabb et al., 2012 (Figure 2) by Brandon S. Tober.
  

