#!/bin/bash
# Helper script to get Python 3.6 on Ubuntu.
echo "Adding deadsnakes repository..."
add-apt-repository ppa:deadsnakes/ppa
echo "Updating apt-get list and installing python3.6"
apt-get update
apt-get install python3.6
echo ""
echo "All done. Check by running 'python3.6 -V'"
echo "To get python3.6 to run with python3 command, you can add the following alias to your ~/.bash_aliases file:"
echo "python3=/usr/bin/python3.6"
echo "Then, run 'source ~./bash_aliases"

