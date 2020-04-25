# LSPARCOM

main14,py, FUNCS_GEN.py, and basic_model.py are used for training the network in python, as explained in https://arxiv.org/abs/2004.09270.
Matlab script create_DS_VAR.m creates the training data (which is too heavy to be uploaded). 

weights750TU.mat is the trained weight file, after training on the TU dataset, as explained in the paper.
weights750BT.mat is the trained weight file, after training on the BT dataset, as explained in the paper.

RUN_CONTEST.m runs reconstruction for the SMLM 2016 contest data, which can be found in http://bigwww.epfl.ch/smlm/datasets/index.html, and calls all other matlab functions.
