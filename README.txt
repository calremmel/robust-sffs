# ******************************************************************************
# Author      : Christopher Remmel (christopher.remmel@dartmouth.edu)
# PI          : Prof. Margaret E. Ackerman
# Project     : P186
# Description : Help file for running the scripts to analyze Fc Array measurements
#		associated with P186
# ******************************************************************************

# Copyright (C) <2019>  <Christopher Remmel>

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

#***********************
Input Files (in the directory data/)
1. 20190904_p186_2A3A.csv
2. 20190904_p186_3A_Only.csv
#***********************

#***********************
Installation

Use conda to install the environment necessary to run these scripts. If you do not have conda installed, you may do so here:
https://docs.anaconda.com/anaconda/install/

From the root directory, execute the following commands:

#-----------------------
conda env create -f environment.yml
conda activate dabl
#-----------------------

The first command need only be run once, and might take a while to execute. The second command you will need to run every time
you wish to use these scripts.

#***********************
All of the remaining commands are to be run from the src/ directory.

To run the feature selection pipeline, execute the following two commands:
#***********************

#-----------------------
1. 
# For 2A3A version:
python interface.py --command=select --config=config_2a3a.yml
# For 3A only version
python interface.py --command=select --config=config_3aOnly.yml
#-----------------------
===> This script performs feature selection and classification to identify vaccine groups using the biophysical measurements
		(1) Feature selection is done using the wrapper method Sequential Forward Floating Selection, with Logistic Regression
		as the wrapped classifier.
		(2) A plot displaying cross-validated accuracy vs number of features is produced.

#-----------------------
2. 
# For 2A3A version:
python interface.py --command=permtest --config=config_2a3a.yml
# For 3A only version
python interface.py --command=permtest --config=config_3aOnly.yml
#-----------------------

===> 
		(1) Permutation tests are performed by shuffling the rows of the class labels, but keeping the rows of feature matrix the same. The permuted data are sent through the same pipeline as were the actual data. This is repeated ten times, independently permuting the class labels every time.
		(2) Robustness is estimated by comparing the accuracies from using actual features to those of using permuted features.

#***********************
For generating remaining figures as in the manuscript, execute the following command:
#***********************

#-----------------------
1. 
# For 2A3A version:
python interface.py --command=evaluate --config=config_2a3a.yml
# For 3A only version
python interface.py --command=evaluate --config=config_3aOnly.yml
#-----------------------
===> This script generates all remaining figures. Run this after after all the above scripts have finished successfully.
		The metrics displayed in these figures are based on Logistic Regression trained using the best feature set identified
		using the above scripts, trained and tested on all samples.

# ******************************************************************************

#-----------------------
System configuration 
#-----------------------
OS  : Ubuntu 18.04.2 LTS
CPU : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
RAM : 8 GB

#-----------------------
Software and packages (version, if specified)
#-----------------------
dependencies:
  - python (3.7)
  - scikit-learn
  - pandas
  - matplotlib (3.1.0)
  - seaborn
  - pyyaml
  - click
  - ipython
  - pip:
    - imbalanced-learn
    - mlxtend

#-----------------------
Functions used by the scripts (in the directory src/)
#-----------------------
01. model.py
02. functions.py