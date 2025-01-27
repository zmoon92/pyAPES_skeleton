# -*- coding: utf-8 -*-
"""
.. module: pyAPES
    :synopsis: APES-model component
.. moduleauthor:: Kersti Haahti

Model framework for 

Created on Tue Oct 02 09:04:05 2018

Note:
    migrated to python3
    - print on same line
    - dict.keys(), but these are iterated after in for-each-loop

References:
Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
forest CO2, H2O and energy flows by a vertically structured forest canopy –
Soil model with separate bryophyte layer. Ecological modelling, 312, pp.385-405.

To call model and run single simulation and read results: see example in sandbox.py
    from tools.iotools import read_results
    from pyAPES import driver
    # for NetCDF-outputs
    outputfile = driver(create_ncf=True, result_file='test.nc')
    results = read_results(outputfile) # opens NetCDF-file using xarray
    
    # for returning results directly
    results = driver(create_ncf=False) # returns dict with integer keys
    results = results[0] # first simulation

Todo:
    
    - make minimal running example (parameterization & example forcing file)
    - make minimal example of handling and plotting outputs using xarray -tools;
      now see tools.iotools.read_forcing for documentation! 
      
"""

import numpy as np
#from matplotlib import pyplot as plt
import time
from copy import deepcopy as copy

from tools.iotools import read_forcing, initialize_netcdf,  write_ncf
from canopy.canopy import CanopyModel
from soil.soil import Soil

from parameters.parametersets import iterate_parameters

import logging


def driver(create_ncf=False,
           result_file=None,
           parametersets={},
           logging_configuration={},
           ):
    """
    Reads parameters, prepares output files, runs model.
    Args:
        create_ncf (bool): results saved to netCDF4 file
        result_file (str): name of result file
        parametersets (dict): parameter sets to overwrite default parameters
    """
    # --- LOGGING ---
    if not logging_configuration:
        from parameters.general import logging_configuration
    from logging.config import dictConfig
    dictConfig(logging_configuration)
    #mpl_logger = logging.getLogger('matplotlib')
    #mpl_logger.setLevel(logging.WARNING)

    # --- READ PARAMETERS ---
    
    # Import general parameters
    from parameters.general import gpara
    # Import canopy model parameters
    from parameters.canopy import cpara
    # Import soil model parameters
    from parameters.soil import spara

    if parametersets == {}:
        Nsim = 1
    else:
        Nsim = parametersets['count']

    default_params = {
            'general': gpara,
            'canopy': cpara,
            'soil': spara
            }

    param_space = [iterate_parameters(parametersets, copy(default_params), count) for count in range(Nsim)]

    logger = logging.getLogger(__name__)

    logger.info('Simulation started. Number of simulations: {}'.format(Nsim))

    # --- FORCING ---
    # Read forcing
    forcing = read_forcing(
        param_space[0]['general']['forc_filename'],
        param_space[0]['general']['start_time'],
        param_space[0]['general']['end_time'],
        dt=param_space[0]['general']['dt']
    )

    # SINGLE SOIL LAYER
#    # read soil moisture and temperature
#    df = read_forcing(gpara['forc_filename'],
#                      gpara['start_time'],
#                      gpara['end_time'],
#                      dt=gpara['dt'],
#                      cols=['Tsoil','Wliq'])
#
#    forcing[['Tsoil','Wliq']] = df[['Tsoil','Wliq']].copy()
#    # set first values as initial conditions
#    spara['heat_model']['initial_condition']['temperature'] = forcing['Tsoil'].iloc[0]
#    spara['water_model']['initial_condition']['volumetric_water_content'] = forcing['Wliq'].iloc[0]

    
    tasks = []

    for k in range(Nsim):
        tasks.append(Model(param_space[k]['general'], param_space[k]['canopy'], param_space[k]['soil'], forcing, nsim=k))
    
    if create_ncf: # outputs to NetCDF-file, returns filename
        timestr = time.strftime('%Y%m%d%H%M')
        if result_file:
            filename = result_file
        else:
            filename = timestr + '_pyAPES_results.nc'

        ncf, _ = initialize_netcdf(
                gpara['variables'],
                Nsim,
                tasks[k].Nsoil_nodes,
                tasks[k].Ncanopy_nodes,
                tasks[k].Nplant_types,
                forcing,
                filepath=gpara['results_directory'],
                filename=filename)

        for task in tasks:
            logger.info('Running simulation number (start time %s): %s' % (
                        time.strftime('%Y-%m-%d %H:%M'), task.Nsim))
            running_time = time.time()
            results = task.run()
            logger.info('Running time %.2f seconds' % (time.time() - running_time))
            write_ncf(nsim=task.Nsim, results=results, ncf=ncf)

            del results

        output_file = gpara['results_directory'] + filename
        logger.info('Ready! Results are in: ' + output_file)
        ncf.close()
        return output_file
    
    else: # returns dictionary of outputs
        running_time = time.time()
        results = {task.Nsim: task.run() for task in tasks}
        
        logger.info('Running time %.2f seconds' % (time.time() - running_time))
        
        return results #, Model # this would return also last Model instance


class Model(object):
    """
    pyAPES - main model class.
    Combines submodels 'CanopyModel' and 'Soil' and handles data-transfer
    between these model components and writing results.
        
    Last edit: SL 17.11.2019
    """
    def __init__(self,
                 gen_para,
                 canopy_para,
                 soil_para,
                 forcing,
                 nsim=0):

        self.dt = gen_para['dt']

        self.Nsteps = len(forcing)
        self.forcing = forcing
        self.Nsim = nsim

        self.Nsoil_nodes = len(soil_para['grid']['dz'])
        self.Ncanopy_nodes = canopy_para['grid']['Nlayers']

        # create soil model instance
        self.soil = Soil(soil_para)

        # create canopy model instance
        # initial delayed temperature and degreedaysum for pheno & LAI-models
        if canopy_para['ctr']['pheno_cycle'] and 'X' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['phenop'].update({'Xo': forcing['X'].iloc[0]})
        if canopy_para['ctr']['seasonal_LAI'] and 'DDsum' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['laip'].update({'DDsum0': forcing['DDsum'].iloc[0]})


        self.canopy_model = CanopyModel(canopy_para, self.soil.grid['dz'])

        self.Nplant_types = len(self.canopy_model.planttypes)

        # initialize structure to save results
        self.results = _initialize_results(gen_para['variables'],
                                       self.Nsteps,
                                       self.Nsoil_nodes,
                                       self.Ncanopy_nodes,
                                       self.Nplant_types)

    def run(self):
        """
        Loops through self.forcing and appends to self.results.
        
        self.forcing variables and units; correspond to uppermost gridpoint:           
            precipitation [m/s]
            air_pressure [Pa]
            air_temperature [degC]
            wind_speed [m/s]
            friction_velocity [m/s]
            h2o[mol/mol]
            co2 [ppm]
            zenith_angle [rad]
            lw_in: Downwelling long wave radiation [W/m2]
            diffPar: Diffuse PAR [W/m2] 
            dirPar: Direct PAR [W/m2]
            diffNir: Diffuse NIR [W/m2] 
            dirNir: Direct NIR [W/m2]
        """

        logger = logging.getLogger(__name__)
        logger.info('Running simulation {}'.format(self.Nsim))
        time0 = time.time()

        #print('RUNNING')
        k_steps=np.arange(0, self.Nsteps, int(self.Nsteps/10))

        for k in range(0, self.Nsteps):

            # --- print progress on screen
            if k in k_steps[:-1]:
                s = str(np.where(k_steps==k)[0][0]*10) + '%'
                print('{0}..'.format(s), end=' ')

            # --- CanopyModel ---
            # run daily loop: updates LAI, phenology and moisture stress ---         
            if self.forcing['doy'].iloc[k] != self.forcing['doy'].iloc[k-1] or k == 0:
                self.canopy_model.run_daily(
                        self.forcing['doy'].iloc[k],
                        self.forcing['Tdaily'].iloc[k])
            
            # compile forcing dict for canopy model: soil_ refers to state of soil model
            canopy_forcing = {
                'wind_speed': self.forcing['U'].iloc[k],            # [m s-1]
                'friction_velocity': self.forcing['Ustar'].iloc[k], # [m s-1]
                'air_temperature': self.forcing['Tair'].iloc[k],    # [deg C]
                'precipitation': self.forcing['Prec'].iloc[k],      # [m s-1]
                'h2o': self.forcing['H2O'].iloc[k],                 # [mol mol-1]
                'co2': self.forcing['CO2'].iloc[k],                 # [ppm]
                'PAR': {'direct': self.forcing['dirPar'].iloc[k],   # [W m-2]
                        'diffuse': self.forcing['diffPar'].iloc[k]},
                'NIR': {'direct': self.forcing['dirNir'].iloc[k],   # [W m-2]    
                        'diffuse': self.forcing['diffNir'].iloc[k]},
                'lw_in': self.forcing['LWin'].iloc[k],              # [W m-2]
                'air_pressure': self.forcing['P'].iloc[k],          # [Pa]
                'zenith_angle': self.forcing['Zen'].iloc[k],        # [rad]
                
                # from soil model
                'soil_temperature': self.soil.heat.T[self.canopy_model.ix_roots],       # [deg C]
                'soil_water_potential': self.soil.water.h[self.canopy_model.ix_roots],  # [m] ?
                'soil_volumetric_water': self.soil.heat.Wliq[self.canopy_model.ix_roots], # [m3 m-3]
                'soil_volumetric_air': self.soil.heat.Wair[self.canopy_model.ix_roots],   # [m3 m-3]
                'soil_pond_storage': self.soil.water.h_pond,                              # [m]  
            }

            canopy_parameters = {
                'soil_depth': self.soil.grid['z'][0],   # [m]
                'soil_hydraulic_conductivity': self.soil.water.Kv[self.canopy_model.ix_roots], # [m s-1]
                'soil_thermal_conductivity': self.soil.heat.thermal_conductivity[0],        # [W m-1 K-1]?
                # SINGLE SOIL LAYER
                # 'state_water':{'volumetric_water_content': self.forcing['Wliq'].iloc[k]},
                #                'state_heat':{'temperature': self.forcing['Tsoil'].iloc[k]}
                'date': self.forcing.index[k]   # pd.datetime
            }

            # call self.canopy_model.run to solve above-ground part
            outputs_canopy, ffloor_flux, ffloor_state, outputs_planttype = self.canopy_model.run(
                dt=self.dt,
                forcing=canopy_forcing,
                parameters=canopy_parameters
            )

            # --- Soil model  ---
            # compile forcing for Soil: potential infiltration and evaporation are at from ground surface
            soil_forcing = {
                'potential_infiltration': ffloor_flux['potential_infiltration'],
                'potential_evaporation': (ffloor_flux['evaporation_soil'] +
                                          ffloor_flux['capillar_rise'] +
                                          ffloor_flux['pond_recharge']),
                'atmospheric_pressure_head': -1.0E6,  # set to large value, because potential_evaporation already account for h_soil
                'ground_heat_flux': -ffloor_flux['ground_heat'],
                'date': self.forcing.index[k]}

            # call self.soil to solve below-ground water and heat flow
            soil_flux, soil_state = self.soil.run(
                    dt=self.dt,
                    forcing=soil_forcing,
                    water_sink=outputs_canopy['root_sink'])


            # --- append results and copy of forcing to self.results
            forcing_output = {
                    'wind_speed': self.forcing['U'].iloc[k],
                    'friction_velocity': self.forcing['Ustar'].iloc[k],
                    'air_temperature': self.forcing['Tair'].iloc[k],
                    'precipitation': self.forcing['Prec'].iloc[k],
                    'h2o': self.forcing['H2O'].iloc[k],
                    'co2': self.forcing['CO2'].iloc[k],
                    'pressure': self.forcing['P'].iloc[k],
                    'par':  self.forcing['dirPar'].iloc[k] + self.forcing['diffPar'].iloc[k],
                    'nir':  self.forcing['dirNir'].iloc[k] + self.forcing['diffNir'].iloc[k],
                    'lw_in': self.forcing['LWin'].iloc[k]
                    }
                
            ffloor_state.update(ffloor_flux)
            soil_state.update(soil_flux)

            self.results = _append_results('forcing', k, forcing_output, self.results)
            self.results = _append_results('canopy', k, outputs_canopy, self.results)
            self.results = _append_results('ffloor', k, ffloor_state, self.results)
            self.results = _append_results('soil', k, soil_state, self.results)
            self.results = _append_results('pt', k, outputs_planttype, self.results)

        print('100%')
        
        ptnames = [pt.name for pt in self.canopy_model.planttypes]
        ptnames.sort()

        self.results = _append_results('canopy', None, {'z': self.canopy_model.z,
                                                        'planttypes': np.array(ptnames)}, self.results)
        
        self.results = _append_results('soil', None, {'z': self.soil.grid['z']}, self.results)

        logger.info('Finished simulation %.0f, running time %.2f seconds' % (self.Nsim, time.time() - time0))

        return self.results


def _initialize_results(variables, Nstep, Nsoil_nodes, Ncanopy_nodes, Nplant_types):
    """
    Creates temporary results dictionary to accumulate simulation results
    SL 12.11.2019: removed if 'date' in dimensions and added option to save planttype profiles
    """

    results = {}

    for var in variables:

        var_name = var[0]
        dimensions = var[2]

        if 'canopy' in dimensions:
            if 'planttype' in dimensions:
                var_shape = [Nstep, Nplant_types, Ncanopy_nodes]            
            else:
                var_shape = [Nstep, Ncanopy_nodes]

        elif 'soil' in dimensions:
            var_shape = [Nstep, Nsoil_nodes]

        elif 'planttype' in dimensions and 'canopy' not in dimensions:
            var_shape = [Nstep, Nplant_types]

        else:
            var_shape = [Nstep]
        
        results[var_name] = np.full(var_shape, np.NAN)
        # print(var_name, var_shape, dimensions)
 
    return results


def _append_results(group, step, step_results, results):
    """
    Adds results from each simulation steps to temporary results dictionary
    """

    results_keys = results.keys()
    step_results_keys = step_results.keys()

    for key in step_results_keys:
        variable = group + '_' + key
        if variable in results_keys:
            if key == 'z' or key == 'planttypes':
                results[variable] = step_results[key]
        
            else:
                results[variable][step] = step_results[key]

    return results

#if __name__ == '__main__':
#
#    from parameters.parametersets import lettosuo_parameters
#    outputfile=driver(create_ncf=True, parametersets=lettosuo_parameters)
#
#    print(outputfile)
