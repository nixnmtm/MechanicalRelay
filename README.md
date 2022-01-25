# Mechanical Relay of residues due to reorganization of protein mechanical coupling network

1. Construction of rigidity graph.
2. Statistical analysis for identification of prominent modes.
3. Mechanical relay - prominent mechanical responses due to substrate/inhibitor dissociation/association.

MechanicalRelay -- Parent directory <br>
|------README -- this file with instructions for installation.<br>
|------	dataset -- the spring constants obtained from structure mechanics statistical learning (apoRT and holoRT).<br>
|------	RT_demo_mechanical_relay.ipynb -- A Jupyter notebook that demonstrate the usage for getting the prominent mechanical responses of RT.<br>
|------	rigidityGraph -- contains the python code<br>
    |------	PDB -- contains the pdb files<br>
    |------	rg -- the root package of rigidity graph module<br>
    	    |------ **MechRelay**  --- the utilities to obtain prominent mechanical responses<br>
    	    |------	utils -- Module for Kolmogorov-Smirnov distribution analysis and plotting.<br>
    	    |------	base.py -- Base module for loading and getting the residue based coupling strength.<br>
    	    |------	core.py -- Sub-class of base. Construction of rigidity graph, spectral decomposition and mode analysis<br>
    	    |------	extras.py -- extra functions for analysis.<br>


Installation:

1. Create a conda environment:

conda create --name rigmat
conda activate rigmat

2. Installation of require packages:

conda install python=3.7 scipy=1.3.0 numpy=1.17.3 matplotlib natsort joblib seaborn pandas jupyter networkx=2.3

3. To use the environment as kernel in Jupyter notebook:

python -m ipykernel install --user --name rigmat --display-name "rigmat"

Chains of spatially contiguous residues are found to exhibit prominent changes in their mechanical rigidity upon substrate binding or dissociation.


References: 

1. Using the code to construct rigidity graph and statistical analysis to obtain the prominent modes, 
please cite<br>
*N. Raj, T. Click, H. Yang and J.-W. Chu, Comput. Struct.Biotechnol. J., 2021, 19, 5309–5320*

2. For identification of prominent mechanical responses of the substrate unbinding, please cite<br>
*N. Raj, T. Click, H. Yang and J.-W. Chu, Chem. Sci. RSC, https://doi.org/10.1039/D1SC06184D*


