# Dependencies and Installation
## Docker
This application is intended to be used with docker. See the Dockerfile.
TODO: add more details.

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

# MySQL Configuration

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
