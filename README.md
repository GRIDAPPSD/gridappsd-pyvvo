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

## Python
This application works in Python 3. It is recommended that you use the latest version, which was 3.6.5 at the time of writing. On Ubuntu, installation of Python 3.6.x requires the addition of [Felix Krull's deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa):
```Shell Session
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
```

## Python packages
There is an included "requirements.txt" file in the "pyvvo" directory. Note that the "gridappsd-python" also has a "requirements.txt" file, so we'll need to install those packages too. Python's pip package manager can be used to easily install packages. Here's an example if you're installing the packages for the entire system:
```Shell Session
$ cd ~/gridappsd-pyvvo/pyvvo
$ sudo -H python3.6 -m pip install -r requirements.txt
$ cd ../gridappsd-python
$ sudo -H python3.6 -m pip install -r requirements.txt
```

# MySQL Configuration
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
