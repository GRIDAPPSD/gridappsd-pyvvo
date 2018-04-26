#!/bin/bash
# Script to install necessary MySQL components, and set up database and user.
# Note that this has only been tested on Ubuntu 16.04
#********************************************************************************
# Install mysql
echo "Installing MySQL. Note root password will be empty."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -q -y install mysql-server
# Configure MySQL to start on boot
update-rc.d mysql defaults
# Change the default socket location
cat <<EOT >> /etc/mysql/my.cnf
# Change default socket location
[mysqld]
socket=/tmp/mysql.sock

[client]
socket=/tmp/mysql.sock
EOT

# Install cmake
echo "Install cmake..."
apt-get -y install cmake
# Install the MySQL Connector/C. Download link is for 6.1.11
echo "Downloading and installing MySQL Connector/C from source..."
CONNECTOR="mysql-connector-c-6.1.11-src"
FNAME="${CONNECTOR}.tar.gz"
wget -O "${FNAME}" "https://dev.mysql.com/get/Downloads/Connector-C/${FNAME}"
tar zxvf "${FNAME}"
cd "${CONNECTOR}"
cmake -G "Unix Makefiles"
make
make install
cd "../"
echo ""
echo "MySQL Connector/C installation complete. Removing ${FNAME} and ${CONNECTOR}"
rm -f ./${FNAME}
rm -rf ./${CONNECTOR}

# Need a symbolic link for GridLAB-D, since it expects the connector to be at /usr/local/mysql-connector-c
echo "Creating simlink from /usr/local/mysql to /usr/local/mysql-connector-c..."
ln -s /usr/local/mysql /usr/local/mysql-connector-c

#********************************************************************************
# Create user and database:
echo "Creating database and user..."
service mysql restart
# Script found on stack overflow and modified:
# https://stackoverflow.com/questions/33470753/create-mysql-database-and-user-in-bash-script
# Looks like original script is here:
# https://raw.githubusercontent.com/saadismail/useful-bash-scripts/master/db.sh

# We'll be creating two databases and users. One for the pyvvo application, and
# another for gridlabd autotests.

# create empty password for database user
DBPASS="pyvvo"
DBPASS2=""
# Database name
DBNAME="pyvvo"
DBNAME2="gridlabd"
# Database user
DBUSER="pyvvo"
DBUSER2="gridlabd"

# Define database commands
CMD1="CREATE DATABASE ${DBNAME};"
CMD2="CREATE DATABASE ${DBNAME2};"
CMD3="CREATE USER '${DBUSER}'@'localhost' IDENTIFIED BY '${DBPASS}';"
CMD4="CREATE USER '${DBUSER}'@'localhost' IDENTIFIED BY '${DBPASS2}';"
CMD5="GRANT ALL PRIVILEGES ON ${DBNAME}.* TO '${DBUSER}'@'localhost';"
CMD6="GRANT ALL PRIVILEGES ON ${DBNAME2}.* TO '${DBUSER2}'@'localhost';"
CMD7="FLUSH PRIVILEGES;"

# If /root/.my.cnf exists then it won't ask for root password
if [ -f /root/.my.cnf ]; then

    mysql -e "${CMD1}"
    mysql -e "${CMD2}"
    mysql -e "${CMD3}"
    mysql -e "${CMD4}"
    mysql -e "${CMD5}"
    mysql -e "${CMD6}"
    mysql -e "${CMD7}"

# If /root/.my.cnf doesn't exist then it'll ask for root password
else
    read -s -p "Please enter MySQL root user password: " rootpasswd
    echo
    mysql -uroot -p${rootpasswd} -e "${CMD1}"
    mysql -uroot -p${rootpasswd} -e "${CMD2}"
    mysql -uroot -p${rootpasswd} -e "${CMD3}"
    mysql -uroot -p${rootpasswd} -e "${CMD4}"
    mysql -uroot -p${rootpasswd} -e "${CMD5}"
    mysql -uroot -p${rootpasswd} -e "${CMD6}"
    mysql -uroot -p${rootpasswd} -e "${CMD7}"
fi

echo "Successfully created database ${DBNAME} and user ${DBUSER} (unless MySQL errors were printed out)."
