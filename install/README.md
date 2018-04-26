Collection of bash scripts for installing application dependencies.
Be sure to check comments at the top of each script before running.

# MySQLSetup.sh:
1. Installs MySQL
2. Does some light configuration (changes default socket, run service on startup)
3. Installs cmake
4. Builds the MySQL Connect/C from source, adds symbolic link for GridLAB-D
5. Creates databases "pyvvo" and "gridlabd"
6. Creates MySQL users "pyvvo" and "gridlabd"

# python36Setup.sh:
1. Adds repository (deadsnakes) which has Python 3.6
2. Installs Python 3.6

# gridlabdPreReqs.sh:
Installs gridlab-d prerequisites automake, libtool, and xerces-c.

# gridlabdBuild.sh:
Helper script to build GridLAB-D from source.
