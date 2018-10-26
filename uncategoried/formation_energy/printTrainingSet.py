#!/usr/bin/env python
#coding: utf8 

import sys
import scipy
import Machine
import cPickle
import time
from numpy import *

	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This script prints the training set "TrainingSet.pkl". Each crystal is separated by the line "--- i ---" and is
# ordered as follows:
#
# --- i --- : index of the crystal i
# Formation energy (eV/atom):  Formation energy of the crystal, given in eV/atom.
# Coordinates: Reduced coordinates of each atom in the unit cell.
# Cell: Unit cell of the crystal, given in (Å).
# Atoms: Atoms in the unit cell of the crystal.
# Representation: The representation of the crystal, used for the learning and the prediction.
# 
# 
# An example of how a crystal is printed:
#
# --- 1807 ---
# Formation energy (eV/atom):  1.0687627526
# Coordinates: 
# 0.0 0.0 0.0
# 0.5 0.5 0.5
# 0.2527 0.7473 0.7473
# 0.2527 0.7473 0.2527
# 0.7473 0.2527 0.7473
# 0.7473 0.7473 0.2527
# 0.2527 0.2527 0.7473
# 0.7473 0.2527 0.2527
# 0.25 0.25 0.25
# 0.75 0.75 0.75
# Cell: 
# 0.0 6.220735 6.220735
# 6.220735 0.0 6.220735
# 6.220735 6.220735 0.0
# Atoms: 
# I K Sn Sn Sn Sn Sn Sn Xe Xe
# Representation: 
# 5 7
# 4 1
# 5 8
# 5 4
#
#---------------------------------------------------------------------------------------------------------------
# The script takes takes the path to "TrainingSet.pkl".
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

periodictable_symbols=asarray([0,'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
                       'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
                       'I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
                       'Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No',
                       'Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Uut','Uuq','Uup','Uuh','Uus','Uuo'])
	
def main():	  
	
	path = str(sys.argv[1])
	
	f = open(path, 'rb')
	inp = cPickle.load(f)
	f.close()
	inp['T'] = inp['T']/inp['N']
	for i in range(len(inp['T'])):
		print('---', i,'---')
		print('Formation energy (eV/atom): ', inp['T'][i])
		print('Coordinates: ')
		for j in range(len(inp['Co'][i])):
			print(inp['Co'][i,j,0],inp['Co'][i,j,1],inp['Co'][i,j,2])
		print('Cell: ')
		for j in range(len(inp['Ce'][i])):
			print(inp['Ce'][i,j,0],inp['Ce'][i,j,1],inp['Ce'][i,j,2])
		print('Atoms: ')
		print(" ".join(str(periodictable_symbols[e]) for e in  inp['Z'][i]))
		print('Representation: ')
		for j in range(len(inp['X'][i])):
			print (inp['X'][i,j,0],inp['X'][i,j,1])


	
if __name__ == "__main__":
	main()
	print('end')
