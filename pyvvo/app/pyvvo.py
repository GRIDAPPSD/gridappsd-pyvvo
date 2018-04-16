'''
This is the 'main' module for the application

Created on Jan 25, 2018

@author: thay838
'''
# Standard library imports:
# Prefer simplejson package
try:
    import simplejson as json
except ImportError:
    import json
import os
import sys

# Get this directory.
thisDir = os.path.dirname(os.path.realpath(__file__))

# pyvvo imports:
from cim import sparqlCIM
from util import db
from glm import modGLM
from genetic import population
import constants as CONST
from util.helper import clock

# If this directory isn't on Python's path, add it
if thisDir not in sys.path:
    sys.path.append(thisDir)
    
def main(fdrid='_9CE150A8-8CC5-A0F9-B67E-BBD8C79D3095'):
    """Main function.
    """
    # Read the config file.
    config = readConfig()
    
    # Get sparqlCIM object, get regulator and capacitor data.
    sparqlObj = sparqlCIM.sparqlCIM(**config['BLAZEGRAPH'])
    reg = sparqlObj.getRegs(fdrid=fdrid)
    cap = sparqlObj.getCaps(fdrid=fdrid)
    
    # Connect to the MySQL database for gridlabd simulations
    dbObj = db.db(**config['GLD-DB'],
                  pool_size=config['GLD-DB-NUM-CONNECTIONS'])
    
    # Clear out the database while testing.
    # TODO: take this out?
    dbObj.dropAllTables()
    
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
    swingMeterName = modelObj.setupModel(starttime=st,
                                         stoptime=et, timezone=tz,
                                         database=config['GLD-DB'],
                                         triplexGroup=CONST.TRIPLEX_GROUP,
                                         powerflowFlag=True)
    
    # Write the base model
    modelObj.writeModel()
    
    # Initialize a clock object for datetimes.
    clockObj = clock(startStr=st, finalStr=et,
                     interval=config['INTERVALS']['OPTIMIZATION'],
                     tzStr=tz)
    
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
                           recordMode='a')
    
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
                                   probabilities=config['PROBABILITIES'])
    
    bestInd = popObj.ga()
    
    print('hoorah')
    
def readConfig():
    """Helper function to read pyvvo configuration file.
    """
    with open(os.path.join(thisDir, 'config.json')) as c:
        config = json.load(c)    
    
    return config

def buildRecorderDicts(energyInterval, powerInterval, voltageInterval, 
                       energyPowerMeter, triplexGroup, recordMode):
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
                              'mode': recordMode
                              }
               },
    'power': {'objType': 'recorder',
              'properties': {'parent': energyPowerMeter,
                             'table': 'power',
                             'interval': powerInterval,
                             'propList': ['measured_real_power',
                                          'measured_reactive_power'],
                             'limit': -1,
                             'mode': recordMode
                            }
               },
    'triplexVoltage': {'objType': 'group_recorder',
                       'properties': {'group': triplexGroup,
                                      'prop': 'measured_voltage_12',
                                      'interval': voltageInterval,
                                      'table': 'triplexVoltage',
                                      'limit': -1,
                                      'complex_part': ['MAG',],
                                      'mode': recordMode
                                      }
                       }
    }
    
    return recorders

if __name__ == '__main__':
    #main(fdrid='_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3')
    main()
    print('yay')