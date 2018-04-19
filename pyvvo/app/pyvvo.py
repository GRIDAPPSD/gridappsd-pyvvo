'''
This is the 'main' module for the application

Created on Jan 25, 2018

@author: thay838
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
#import traceback

# Get this directory.
THISDIR = os.path.dirname(os.path.realpath(__file__))

# If this directory isn't on Python's path, add it
if THISDIR not in sys.path:
    sys.path.insert(0, THISDIR)
    
# get the config file
CONFIGFILE = os.path.join(THISDIR, 'config.json')

# pyvvo imports:
import sparqlCIM
import db
import modGLM
import population
import constants as CONST
from helper import clock
    
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
    log.info('Load nominal voltage data pulled from blazegraph.')
    
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
                            vSource=config['FEEDER']['SUBSTATION-VOLTAGE'],
                            triplexGroup=CONST.TRIPLEX_GROUP,
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
                           triplexGroup=CONST.TRIPLEX_GROUP,
                           recordMode='a',
                           query_buffer_limit=config['GLD-DB-OTHER']['QUERY_BUFFER_LIMIT'])
    
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
                                   costs=config['COSTS'],
                                   probabilities=config['PROBABILITIES'],
                                   gldPath=config['GLD-PATH'],
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

if __name__ == '__main__':
    #main(fdrid='_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3')
    main()