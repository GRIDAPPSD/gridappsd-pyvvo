# Dependencies and Installation
## Submodule: "gridappsd-python"
In order to interact with the GridAPPS-D platform, the pyvvo application depends on the ["gridappsd-python"](https://github.com/GRIDAPPSD/gridappsd-python) repository and its package, "gridappsd." This repository has been included in this one as a [git submodule](https://git-scm.com/docs/git-submodule). After cloning the gridappsd-pyvvo repository, you'll need to change directories into the top level "gridappsd-python" directory and enter two git commands:
```Shell Session
$ git submodule init
$ git submodule update
```

Alternatively, while cloning the gridappsd-pyvvo repository, you can pass the `--recurse-submodules` option to the `git clone` command like so:
```Shell Session
$ git clone --recurse-submodules https://github.com/GRIDAPPSD/gridappsd-pyvvo.git
```

The simplest way to keep the "gridappsd-python" repository up to date within this repository is by running:
```Shell Session
$ git submodule update --remote gridappsd-python
```

## Git Large File Storage (LFS)
This repository contains some .xml and .glm files which are quite large. To keep the repository light, these files are tracked with git lfs. 
To install [source](https://github.com/git-lfs/git-lfs/wiki/Installation):

### Windows
1. Navigate [here](https://git-lfs.github.com), download the installer, and run it.
2. From a git bash terminal, navigate into the head of this repository and `git lfs install`

### Ubuntu
1. Add the git-core repository from the terminal: `sudo apt-add-repository ppa:git-core/ppa`
2. Download and run the install script: `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
3. Install: `sudo apt-get install git-lfs`
4. Activate: `git lfs install`

## Python
This application works in Python 3. It is recommended that you use the latest version, which was 3.6.5 at the time of writing. On Ubuntu, installation of Python 3.6.x requires the addition of [Felix Krull's deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa):
```Shell Session
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
```

There's a helper script to perform this: install/python36Setup.sh.

## Python packages
There is an included "requirements.txt" file in the "pyvvo" directory. Note that the "gridappsd-python" also has a "requirements.txt" file, so we'll need to install those packages too. Python's pip package manager can be used to easily install packages. Here's an example if you're installing the packages for the entire system:
```Shell Session
$ cd ~/gridappsd-pyvvo/pyvvo
$ sudo -H python3.6 -m pip install -r requirements.txt
$ cd ../gridappsd-python
$ sudo -H python3.6 -m pip install -r requirements.txt
```

## Helper shell scripts
In the top level "install" directory, there are helper scripts for installing MySQL, GridLAB-D, Python3.6, etc.
MySQL installtion + configuration should be performed before GridLAB-D.

# MySQL Configuration
## Installation
You can use the script install/MySQLSetup.sh to install MySQL and the MySQL Connector/C. This script also does
some helpful configuration for connecting with GridLAB-D and the pyvvo application.

## InnoDB
Ensure that the MySQL global variables `innod_db_buffer_pool_size` and `innodb_buffer_pool_instances` are set adequately high.
On Brandon's Windows machine, `innod_db_buffer_pool_size=4G` and `innodb_buffer_pool_instances=16`

These settings can be configured via the options file. On Windows:
1. Run "services.msc" (either from start menu search box or using the "run" utility)
2. Find MySQL in the list, _e.g._ MySQL57
3. Right click on the row with MySQL in it, select "Properties"
4. Under "Path to executable" locate the "--defaults-file" input
5. Open the "--defaults-file," search for "innodb"
6. Alter necessary defaults.
7. Restart MySQL.

Alternatively, these setting can be tweaked in MySQL Workbench the "Options File" button under the "Instance" heading in the "Navigator" pane (on the left edge of the program).

More details (and commands for Linux/Unix) can be found [here](https://dev.mysql.com/doc/refman/5.7/en/innodb-buffer-pool-resize.html)
