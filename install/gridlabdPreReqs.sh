#!/bin/bash
# Simple script to install pre-reqs for building GridLAB-D
# The gridlab-d repository should already be cloned so we can install xerces.
echo "Installing automake and libtool via apt-get..."
apt-get update
apt-get -y install automake
apt-get -y install libtool

echo "Installing xerces from gridlab-d third_party folder..."
cd "gridlab-d/third_party"
tar zxvf "xerces-c-3.1.1.tar.gz"
cd "xerces-c-3.1.1"
./configure
make
make install
