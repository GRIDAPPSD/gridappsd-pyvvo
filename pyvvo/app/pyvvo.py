"""
This is the 'main' module for the application

Created on Jan 25, 2018

@author: thay838
"""

'''
# Standard library imports:
# Prefer simplejson package
try:
    import simplejson as json
    from simplejson.errors import JSONDecodeError
except ImportError:
    import json
    import json.JSONDecodeError as JSONDecodeError
    
import os
import sys
import logging
import copy
#import traceback

# pyvvo imports:
import sparqlCIM
import db
import modGLM
import population
import constants as CONST
from helper import clock

# Get this directory.
THISDIR = os.path.dirname(os.path.realpath(__file__))

# If this directory isn't on Python's path, add it
if THISDIR not in sys.path:
    sys.path.insert(0, THISDIR)
    
# get the config file
CONFIGFILE = os.path.join(THISDIR, 'config.json')
'''
# gridappsd-python imports
from gridappsd import GridAPPSD

# Standard library
import logging as log

# pyvvo
import sparql_queries

# Logging.
log.basicConfig(level=log.INFO)

def main():
    # Current, as of 04/19: 
    # 8500: _4F76A5F9-271D-9EB8-5E31-AA362D86F2C3
    # R2: _0663ADF2-FC00-45BE-858E-50B3D1D01696
    
    # fdrid='_0663ADF2-FC00-45BE-858E-50B3D1D01696'
    # fdrid='_9CE150A8-8CC5-A0F9-B67E-BBD8C79D3095'
    
    #define VSOURCE=66395.3
    #include "test8500_base.glm";
    
    #define VSOURCE=57735.0
    #include "testR2_base.glm";
    
    """Main function.
    """
    # Read the config file.
    try:
        config = readConfig()
    except JSONDecodeError as err:
        print('Config file, {}, not properly formatted!'.format(CONFIGFILE),
              file=sys.stderr)
        print(err, file=sys.stderr)
        print('Exiting...', file=sys.stderr)
        exit(1)
    
    # Setup the log.
    log = setupLog(logConfig=config['LOG'])
    log.info('Configuration file read, log configured.')
    
    # Get sparqlCIM object, get regulator and capacitor data.
    sparqlObj = sparqlCIM.sparqlCIM(**config['BLAZEGRAPH'])
    reg = sparqlObj.getRegs(fdrid=config['FEEDER']['ID'])
    cap = sparqlObj.getCaps(fdrid=config['FEEDER']['ID'])
    log.info('Regulator and Capacitor information pulled from blazegraph.')
    
    # Get dictionary of loads and their nominal voltages
    loadV = sparqlObj.getLoadNomV(fdrid=config['FEEDER']['ID'])
    swingV = sparqlObj.getSwingVoltage(fdrid=config['FEEDER']['ID'])
    log.info('Load and swing bus nominal voltage data pulled from blazegraph.')
    
    # Get dictionary of load measurements
    loadM = sparqlObj.getLoadMeasurements(fdrid=config['FEEDER']['ID'])
    log.info('Load measurement data pulled from blazegraph.')
    
    # Ensure we have a measurement for all loads.
    # TODO: Eventually we should have a way to handle unmeasured loads.
    for loadType in loadV:
        for m in loadV[loadType]['meters']:
            if m not in loadM:
                # If we're missing it, throw an error
                raise UserWarning('Meter {} is not being "measured"!'.format(m))
            
    log.info('Confirmed that all EnergyConsumers have measurements.')
         
    # Connect to the MySQL database for gridlabd simulations
    dbObj = db.db(**config['GLD-DB'],
                  pool_size=config['GLD-DB-OTHER']['NUM-CONNECTIONS'])
    log.info('Connected to MySQL database for GridLAB-D simulation output.')
    
    # Clear out the database while testing.
    # TODO: take this out?
    dbObj.dropAllTables()
    log.warning('All tables dropped in {}'.format(config['GLD-DB']['database']))
    
    outDir = config['PATHS']['outDir']
    baseModel = 'test.glm'
    baseOut = os.path.join(outDir, baseModel)
    # Get a modGLM model to modify the base model.
    modelObj = modGLM.modGLM(pathModelIn=config['PATHS']['baseModel'],
                             pathModelOut=baseOut
                            )
    
    # Set up the model to run.
    st = '2016-01-01 00:00:00'
    et = '2016-01-01 01:00:00'
    tz = 'PST+8PDT'
    swingMeterName = \
        modelObj.setupModel(starttime=st,
                            stoptime=et, timezone=tz,
                            database=config['GLD-DB'],
                            powerflowFlag=True,
                            vSource=swingV,
                            #vSource=config['FEEDER']['SUBSTATION-VOLTAGE'],
                            triplexGroup=CONST.LOADS['triplex']['group'],
                            triplexList=loadV['triplex']['meters']
                            )
    
    # Write the base model
    modelObj.writeModel()
    log.info('Base GridLAB-D model configured.')
    
    # Initialize a clock object for datetimes.
    clockObj = clock(startStr=st, finalStr=et,
                     interval=config['INTERVALS']['OPTIMIZATION'],
                     tzStr=tz)
    log.info('Clock object initialized')
    
    # Build dictionary of recorder definitions which individuals in the
    # population will add to their model. We'll use the append record mode.
    # This can be risky! If you're not careful about clearing the database out
    # between subsequent test runs, you can write duplicate rows.
    recorders = \
        buildRecorderDicts(energyInterval=config['INTERVALS']['OPTIMIZATION'],
                           powerInterval=config['INTERVALS']['SAMPLE'],
                           voltageInterval=config['INTERVALS']['SAMPLE'],
                           energyPowerMeter=swingMeterName,
                           triplexGroup=CONST.LOADS['triplex']['group'],
                           recordMode='a',
                           query_buffer_limit=config['GLD-DB-OTHER']['QUERY_BUFFER_LIMIT'])
    
    # Convert costs from fraction of nominal voltage to actual voltage
    costs = copy.copy(config['COSTS'])
    costs['undervoltage']['limit'] = (costs['undervoltage']['limit']
                                      * loadV['triplex']['v'])
    costs['overvoltage']['limit'] = (costs['overvoltage']['limit']
                                     * loadV['triplex']['v'])
    
    # Initialize a population.
    # TODO - let's get the 'inPath' outta here. It's really just being used for
    # model naming, and we may as well be more explicit about that.
    popObj = population.population(strModel=modelObj.strModel,
                                   numInd=config['GA']['INDIVIDUALS'],
                                   numGen=config['GA']['GENERATIONS'],
                                   numModelThreads=config['GA']['THREADS'],
                                   recorders=recorders,
                                   dbObj=dbObj,
                                   starttime=clockObj.start_dt,
                                   stoptime=clockObj.stop_dt,
                                   timezone=tz,
                                   inPath=modelObj.pathModelIn,
                                   outDir=outDir,
                                   reg=reg, cap=cap,
                                   costs=costs,
                                   probabilities=config['PROBABILITIES'],
                                   gldInstall=config['GLD-INSTALLATION'],
                                   randomSeed=config['RANDOM-SEED'],
                                   log=log)
    
    log.info('Population object initialized.')
    
    bestInd = popObj.ga()
    
    print(bestInd)
    print('hoorah')


def readConfig():
    """Helper function to read pyvvo configuration file.
    """
    with open(CONFIGFILE) as c:
        config = json.load(c)    
    
    return config


def buildRecorderDicts(energyInterval, powerInterval, voltageInterval, 
                       energyPowerMeter, triplexGroup, recordMode,
                       query_buffer_limit):
    """Helper function to construct dictionaries to be used by individuals to
    add recorders to their own models.
    
    Note that the returned dictionary will more or less be directly passed to
    a genetic.individual object, and subsequently passed to the appropriate
    method in glm.modGLM.
    
    We could add custom table definitions in the future, but why?
    """
    recorders = {
    'energy': {'objType': 'recorder',
               'properties': {'parent': energyPowerMeter,
                              'table': 'energy',
                              'interval': energyInterval,
                              'propList': ['measured_real_energy',],
                              'limit': -1,
                              'mode': recordMode,
                              'query_buffer_limit': query_buffer_limit
                              }
               },
    'power': {'objType': 'recorder',
              'properties': {'parent': energyPowerMeter,
                             'table': 'power',
                             'interval': powerInterval,
                             'propList': ['measured_real_power',
                                          'measured_reactive_power'],
                             'limit': -1,
                             'mode': recordMode,
                             'query_buffer_limit': query_buffer_limit
                            }
               },
    'triplexVoltage': {'objType': 'recorder',
                       'properties': {'group': triplexGroup,
                                      'propList': ['measured_voltage_1.mag',
                                                   'measured_voltage_2.mag'],
                                      'interval': voltageInterval,
                                      'table': 'triplexVoltage',
                                      'limit': -1,
                                      'mode': recordMode,
                                      'query_buffer_limit': query_buffer_limit
                                      }
                       }
    }
    
    return recorders


def setupLog(logConfig):
    """Helper to setup the log.
    
    INPUTS:
    logConfig -- sub-dictionary pulled from config['LOG'].
        keys: FILE, MODE, LEVEL, FORMAT
    
    OUTPUTS:
    logging.Logger object
    """
    
    # Set up the log.
    log = logging.getLogger(__name__)
    # Set log level.
    level = getattr(logging, logConfig['LEVEL'].upper())
    log.setLevel(level)
    # Create file handler for log.
    f_h = logging.FileHandler(filename=logConfig['FILE'],
                              mode=logConfig['MODE'])
    # Set its level.
    f_h.setLevel(level)
    # Create formatter for file handler. Use date format in constants module.
    formatter = logging.Formatter(logConfig['FORMAT'],
                                  datefmt=CONST.DATE_FMT)
    # Attach the formatter to the file handler.
    f_h.setFormatter(formatter)
    # Attach the file handler to the log
    log.addHandler(f_h)
    
    return log


def get_model_id(gridappsd_object, model_name):
    """Given a model's name, get it's ID from the platform."""
    # Get model information.
    log.debug('Calling GridAPPSD.query_model_info.')
    result = gridappsd_object.query_model_info()

    #
    if not result['responseComplete']:
        s = ('GridAPPSD.query_model_info responseComplete field not return '
             'True.')
        log.error(s)
        raise UserWarning(s)
    else:
        log.debug('GridAPPSD.query_model_info call successful.')

    # Loop over the models until we find our model_name, and get its ID.
    model_id = None
    for model in result['data']['models']:
        if model['modelName'] == model_name:
            model_id = model['modelId']

    log.debug('Model ID for {} is {}'.format(model_name, model_id))

    if model_id is None:
        s = 'Could not find the model ID for {}.'.format(MODEL)
        log.error(s)
        raise UserWarning(s)

    return model_id


def get_all_model_data(gridappsd_object, model_id):
    """Helper to pull all requisite model data for pyvvo.

    This includes: voltage regulators, capacitors, load nominal
        voltages, swing bus voltage, and load measurement data.
    """
    # Define dictionary for each type of data.
    data_list = [
        {'type': 'voltage_regulator',
         'query_string': sparql_queries.REGULATOR_QUERY,
         'parse_function': sparql_queries.parse_regulator_query
         },
        {'type': 'capacitor',
         'query_string': sparql_queries.CAPACITOR_QUERY,
         'parse_function': sparql_queries.parse_capacitor_query
         },
        {'type': 'load_nominal_voltage',
         'query_string': sparql_queries.LOAD_NOMINAL_VOLTAGE_QUERY,
         'parse_function': sparql_queries.parse_load_nominal_voltage_query
         },
        {'type': 'swing_voltage',
         'query_string': sparql_queries.SWING_VOLTAGE_QUERY,
         'parse_function': sparql_queries.parse_swing_voltage_query
         },
        {'type': 'load_measurements',
         'query_string': sparql_queries.LOAD_MEASUREMENTS_QUERY,
         'parse_function': sparql_queries.parse_load_measurements_query
         },
    ]

    # Initialize return.
    out = {}

    # Loop over the data.
    for data_dict in data_list:
        out[data_dict['type']] = query_and_parse(
            gridappsd_object=gridappsd_object,
            query_string=data_dict['query_string'].format(fdrid=model_id),
            parse_function=data_dict['parse_function'],
            log_string=data_dict['type'].replace('_', ' '))

    return out


def query_and_parse(gridappsd_object, query_string, parse_function,
                    log_string):
    # Get data.
    data = gridappsd_object.query_data(query_string)
    log.info('Retrieved {} data.'.format(log_string))

    # Parse data.
    data_parsed = parse_function(data['data']['results']['bindings'])
    log.info('Parsed {} data'.format(log_string))

    return data_parsed


if __name__ == '__main__':
    #main(fdrid='_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3')
    #main()
    # For development, hard-code this machine's internal IP.
    IP = '192.168.0.33'
    # For now, hard-code GridAPPSD port. Later, get from environment
    # variable.
    PORT = 61613
    gridappsd_object = GridAPPSD(address=('192.168.0.33', 61613))

    # We'll be using the 8500 node feeder.
    MODEL = 'ieee8500'

    # Get ID for feeder.
    model_id = get_model_id(gridappsd_object=gridappsd_object,
                            model_name=MODEL)
    log.info('Retrieved model ID.')

    # Get all relevant model data.
    model_data = get_all_model_data(gridappsd_object, model_id)
    pass
