#!/bin/bash
# Script to get GridLAB-D built.
# Assumptions/notes:
# 1) Make sure you run this script from the directory it resides in
# 2) The gridlabd repository should be cloned and exist in ./gridlab-d
# 3) MySQL should be setup and running (use MySQLSetup.sh)
# 4) Dependencies xerces, automake, and libtool have been isntalled (use gridlabdPreReqs.sh)

# Build gridlab-d
echo "Installing gridlab-d into ./gridlabd/builds..."
cd "gridlab-d"
autoreconf -isf
./configure --prefix=$PWD/builds --with-mysql=/usr/local/mysql --enable-silent-rules 'CFLAGS=-g -O0 -w' 'CXXFLAGS=-g -O0 -w' 'LDFLAGS=-g -O0 -w'
make
make install

echo "GridLAB-D has been built and installed. Recomment running the auto-tests."
echo "You'll also need to setup the following environment variables when running:"
echo "(Be sure to substitute ./builds with a full path)"
echo "(Note the pyvvo application only needs the path to ./builds in its config.json file)"
echo "  1) Add ./builds/bin to your path."
echo "  2) GLPATH=\"./builds/lib/gridlabd:./builds/share/gridlabd\""
echo "  3) CXXFLAGS=\"-I./builds/share/gridlabd\""
