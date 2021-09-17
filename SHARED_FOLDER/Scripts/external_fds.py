import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc3 as mc
import tga_exp

def run_fds(infile):
	os.system('cmd /c "fds ' + infile + '.fds"')

def input(chid, theta):

	hr = 60,
	A = [6.14, 10**theta[0], 10**theta[3], 10**theta[6]],
	E = [2.35e4, theta[1], theta[4], theta[7]],
	nu = [0, theta[2], theta[5], theta[8]],
	rho = [520, (520*theta[2]), (520*theta[2]*theta[5]), (520*theta[2]*theta[5]*theta[8])]

	HEAD = '&HEAD CHID = \'' + chid + '\' /'
	MISC = '&MISC SOLID_PHASE_ONLY=.TRUE. /'
	MESH = '&MESH IJK = 3, 3, 3, XB = -0.15, 0.15, -0.15, 0.15, 0.0, 0.3 /'
	TIME = '&TIME T_END=3600., WALL_INCREMENT = 1., DT = 0.05 /'
	SPEC = '&SPEC ID = \'METHANE\' /'
	VENT = '&VENT XB = -0.05, 0.05, -0.05, 0.05, 0.0, 0.0, SURF_ID = \'SAMPLE\' /'
	SURF = '&SURF ID = \'SAMPLE\', TGA_ANALYSIS = .TRUE., TGA_HEATING_RATE = ' + str(hr) + ', MATL_ID(1,1) = \'LB\', MATL_ID(1,2) = \'MOISTURE\', MATL_MASS_FRACTION(1,:) = 0.98,0.02, THICKNESS = 0.001, CELL_SIZE_FACTOR = 0.1, STRETCH_FACTOR(1) = 1. /'
	DUMP = '&DUMP DT_DEVC = 1, SUPPRESS_DIAGNOSTICS =  /'
	TAIL = '&TAIL /'
	DEVC = []
	MATL = []
	IFILE = []

	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL TEMPERATURE\', ID = \'temp\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'BACK WALL TEMPERATURE\', ID = \'back_temp\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'MASS FLUX\', SPEC_ID = \'METHANE\', ID=\'MF\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL THICKNESS\', ID = \'thick\' /' + '\n')

	MATL.append('&MATL ID = \'MOISTURE\', N_REACTIONS = 1, A(1) = ' + str(A[0][0]) + ', E(1) = ' + str(E[0][0]) + ', EMISSIVITY = 0.9, DENSITY = 1000, SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1.0) + ', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'LB\', N_REACTIONS = 1, A(1) = ' + str(A[0][1]) + ', E(1) = ' + str(E[0][1]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[0]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0][1]) + ', NU_MATL = ' + str(nu[0][1]) + ', MATL_ID = \'LB_int\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'LB_int\', N_REACTIONS = 1, A(1) = ' + str(A[0][2]) + ', E(1) = ' + str(E[0][2]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[1]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0][2]) + ', NU_MATL = ' + str(nu[0][2]) + ', MATL_ID = \'LB_char1\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'LB_char1\', N_REACTIONS = 1, A(1) = ' + str(A[0][3]) + ', E(1) = ' + str(E[0][3]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[2]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0][3]) + ', NU_MATL = ' + str(nu[0][3]) + ', MATL_ID = \'LB_char2\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'LB_char2\', EMISSIVITY = 0.9, DENSITY = ' + str(rho[3]) + ', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')

	IFILE.append(HEAD)
	IFILE.append(MESH)
	IFILE.append(TIME)
	IFILE.append(MISC)
	IFILE.append(SPEC)
	IFILE.append(MATL)
	IFILE.append(SURF)
	IFILE.append(VENT)
	IFILE.append(DUMP)
	IFILE.append(DEVC)
	IFILE.append(TAIL)

	comp_file = chid + '.fds'
	with open(comp_file, mode = 'w', encoding='utf-8') as f:
		for lines in IFILE:
			f.write(''.join(str(line) for line in lines))
			f.write('\n')
		f.close
	return

def read_fds(chid):
	data = pd.read_csv(chid + '_tga.csv', skiprows=1)
	data = data.set_index('Temp')
	data = data.reindex(data.index.union(np.arange(50, 650, 0.5)))
	data = data.interpolate(method='index')
	data = data.loc[np.arange(50, 650, 0.5)]

	data_out = data['Total MLR']

	return data_out