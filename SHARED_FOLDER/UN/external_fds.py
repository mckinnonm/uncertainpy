import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

def run_fds(infile):
	
	infile = str(infile)+'.fds'
	ret = os.system("fds " + infile)
	# ret = os.system("gnome-terminal -e 'fds " + infile + "' >/dev/null 2>&1")
	#subprocess.run(['fds ', infile + '.fds'])
	return ret

# *** tga_input written specifically for PMMA ***
def tga_input(chid, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6):

	hr = 10
	A = [10**theta_1, 10**theta_4]
	E = [10000*theta_2, 10000*theta_5]
	nu = [theta_3, theta_6]
	rho = [1160, 1160, 1160] # density does not change for non-charring polymers

	HEAD = '&HEAD CHID = \'' + chid + '\' /'
	MISC = '&MISC SOLID_PHASE_ONLY=.TRUE. /'
	MESH = '&MESH IJK = 3, 3, 3, XB = -0.15, 0.15, -0.15, 0.15, 0.0, 0.3 /'
	TIME = '&TIME T_END=3600., WALL_INCREMENT = 1., DT = 0.05 /'
	SPEC = '&SPEC ID = \'METHANE\' /'
	VENT = '&VENT XB = -0.05, 0.05, -0.05, 0.05, 0.0, 0.0, SURF_ID = \'SAMPLE\' /'
	SURF = '&SURF ID = \'SAMPLE\', TGA_ANALYSIS = .TRUE., TGA_HEATING_RATE = ' + str(hr) + ', MATL_ID(1,1) = \'PMMA\', MATL_MASS_FRACTION(1,:) = 1.0, THICKNESS = 0.001, CELL_SIZE_FACTOR = 0.1, STRETCH_FACTOR(1) = 1. /'
	DUMP = '&DUMP DT_DEVC = 1, SUPPRESS_DIAGNOSTICS = .TRUE. /'
	TAIL = '&TAIL /'
	DEVC = []
	MATL = []
	IFILE = []

	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL TEMPERATURE\', ID = \'temp\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'BACK WALL TEMPERATURE\', ID = \'back_temp\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'MASS FLUX\', SPEC_ID = \'METHANE\', ID=\'MF\' /' + '\n')
	DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL THICKNESS\', ID = \'thick\' /' + '\n')

	MATL.append('&MATL ID = \'PMMA\', N_REACTIONS = 1, A(1) = ' + str(A[0]) + ', E(1) = ' + str(E[0]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[0]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0]) + ', NU_MATL = ' + str(nu[0]) + ', MATL_ID = \'PMMA_int\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'PMMA_int\', N_REACTIONS = 1, A(1) = ' + str(A[1]) + ', E(1) = ' + str(E[1]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[1]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[1]) + ', NU_MATL = ' + str(nu[1]) + ', MATL_ID = \'PMMA_char\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	# MATL.append('&MATL ID = \'PMMA_char1\', N_REACTIONS = 1, A(1) = ' + str(A[0][3]) + ', E(1) = ' + str(E[0][3]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[2]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0][3]) + ', NU_MATL = ' + str(nu[0][3]) + ', MATL_ID = \'LB_char2\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
	MATL.append('&MATL ID = \'PMMA_char\', EMISSIVITY = 0.9, DENSITY = ' + str(rho[2]) + ', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')

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

def cone_input(chid, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9):

	hf = 25
	A = [6.14, 10**theta_1, 10**theta_4, 10**theta_7],
	E = [2.35e4, theta_2, theta_5, theta_8],
	nu = [0, theta_3, theta_6, theta_9],
	rho = [520, (520*theta_3), (520*theta_3*theta_6), (520*theta_3*theta_6*theta_9)]

	HEAD = '&HEAD CHID = \'' + chid + '\' /'
	MISC = '&MISC SOLID_PHASE_ONLY=.TRUE. /'
	MESH = '&MESH IJK = 3, 3, 3, XB = -0.15, 0.15, -0.15, 0.15, 0.0, 0.3 /'
	TIME = '&TIME T_END=1800., WALL_INCREMENT = 1., DT = 0.05 /'
	SPEC = '&SPEC ID = \'METHANE\' /'
	VENT = '&VENT XB = -0.05, 0.05, -0.05, 0.05, 0.0, 0.0, SURF_ID = \'SAMPLE\' /' + '\n'
	VENT.append('&VENT MB = \'XMIN\', SURF_ID = \'OPEN\' /' + '\n')
	VENT.append('&VENT MB = \'XMAX\', SURF_ID = \'OPEN\' /' + '\n')
	VENT.append('&VENT MB = \'YMIN\', SURF_ID = \'OPEN\' /' + '\n')
	VENT.append('&VENT MB = \'YMAX\', SURF_ID = \'OPEN\' /' + '\n')
	VENT.append('&VENT MB = \'ZMAX\', SURF_ID = \'OPEN\' /' + '\n')

	SURF = '&SURF ID = \'SAMPLE\', EXTERNAL_FLUX = ' + str(hf) + ', MATL_ID(1,1) = \'LB\', MATL_ID(1,2) = \'MOISTURE\', MATL_MASS_FRACTION(1,:) = 0.98,0.02, THICKNESS = 0.001, CELL_SIZE_FACTOR = 0.1, STRETCH_FACTOR(1) = 1. /'
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

def read_tga_fds(chid):
	data = pd.read_csv(chid + '_tga.csv', skiprows=1)
	data = data.set_index('Temp')
	data = data.reindex(data.index.union(np.arange(50, 650, 0.5)))
	data = data.interpolate(method='index')
	data = data.loc[np.arange(50, 650, 0.5)]

	nmass = data['Total Mass'] 
	mlr = data['Total MLR'] 
	temp = data.index
	# This should be changed back to Total MLR if the main file changes

	return temp, mlr

def read_cone_fds(chid):
	data = pd.read_csv(chid + '_tga.csv', skiprows=1)
	data = data.set_index('Temp')
	data = data.reindex(data.index.union(np.arange(50, 650, 0.5)))
	data = data.interpolate(method='index')
	data = data.loc[np.arange(50, 650, 0.5)]

	nmass = data['Total Mass'] 
	mlr = data['Total MLR'] 
	temp = data.index
	# This should be changed back to Total MLR if the main file changes

	return temp, mlr
