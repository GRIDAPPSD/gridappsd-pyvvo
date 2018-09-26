# NOTE: Some of this is borrowed from Dockerfile.gridappsd_base
# Use Ubuntu 18.04 LTS
FROM ubuntu:bionic

# Setup some environment variables for installation.
# MSCC --> MySQL Connector/C
# TODO: It seems CZMQ is on 4.1.1 and ZMQ is on 4.1.4 - consider upgrading.
# All libs are going into /pyvvo/lib
# EXCEPT MySQL Connector/C, which will go into /usr/local/mysql/lib
# FNCS_LOG settings
# Environment variables for gridlab-d
# Packages which will be installed to build software then removed later.
ENV PYVVO=/pyvvo
ENV MSCC_VERSION=6.1.11 \
    XERCES_VERSION=3.2.0 \
    CZMQ_VERSION=3.0.2 \
    ZMQ_VERSION=4.0.2 \
    TEMP_DIR=/tmp/source \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYVVO}/lib:/usr/local/mysql/lib \
    FNCS_LOG_FILE=yes \
    FNCS_LOG_STDOUT=yes \
    FNCS_LOG_TRACE=yes \
    FNCS_LOG_LEVEL=DEBUG1 \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd \
    CXXFLAGS=-I${PYVVO}/share/gridlabd \
    PACKAGES="autoconf automake cmake g++ gcc git git-lfs libtool make wget"

# Create temporary directory for all installations
RUN mkdir ${TEMP_DIR}

# Work out of the temporary directory
WORKDIR ${TEMP_DIR}

# Install necessary packages, get Python's pip upgraded
RUN apt-get update && apt-get -y install \
    ${PACKAGES} \
    mysql-server \
    python3.7 \
    python3-pip \
    && rm -rf /var/lib/opt/lists/* \
    && git lfs install \
    && python3.7 -m pip install pip --upgrade \
# ----------------------------------------------------
# Configure MySQL
# ----------------------------------------------------
# Setup innodb buffer and create database and user for GridLAB-D
# NOTE: For some unknown reason, the socket location change is being
# overwritten and won't work.
    && echo "" >> /etc/mysql/my.cnf \
    && echo "[mysqld]" >> /etc/mysql/my.cnf \
    && echo "innodb_buffer_pool_size=4G" >> /etc/mysql/my.cnf \
    && echo "innodb_buffer_pool_instances=16" >> /etc/mysql/my.cnf \
    && service mysql start \
    && mysql -uroot -e "CREATE DATABASE gridlabd;" \
    && mysql -uroot -e "CREATE USER 'gridlabd'@'localhost' IDENTIFIED BY '';" \
    && mysql -uroot -e "GRANT ALL PRIVILEGES ON gridlabd.* to 'gridlabd'@'localhost';" \
    && mysql -uroot -e "FLUSH PRIVILEGES;" \
    && service mysql stop \
#    && echo "socket=/tmp/mysql.sock" >> /etc/mysql/my.cnf \
#    && echo "" >> /etc/mysql/my.cnf \
#    && echo "[client]" >> /etc/mysql/my.cnf \
#    && echo "socket=/tmp/mysql.sock" >> /etc/mysql/my.cnf
# ----------------------------------------------------
# Build MySQL Connector/C, create MySQL simlink for
# GridLAB-D, build ZeroMQ, CZMQ, FNCS, and GridLAB-D
# Finally, clone gridappsd-pyvvo and clean up.
# ----------------------------------------------------
# MySQL Connector/C
    && wget "https://dev.mysql.com/get/Downloads/Connector-C/mysql-connector-c-${MSCC_VERSION}-src.tar.gz" \
    && tar -zxf "mysql-connector-c-${MSCC_VERSION}-src.tar.gz" \
    && cd "mysql-connector-c-${MSCC_VERSION}-src" \
    && cmake -G "Unix Makefiles" -DCMAKE_C_FLAGS="-w" -DCMAKE_CXX_FLAGS="-w"\
    && make \
    && make install \
    && ln -s /usr/local/mysql /usr/local/mysql-connector-c \
# ZeroMQ
    && cd "${TEMP_DIR}" \
    && wget http://download.zeromq.org/zeromq-${ZMQ_VERSION}.tar.gz \
    && tar -xzf zeromq-${ZMQ_VERSION}.tar.gz \
    && cd zeromq-${ZMQ_VERSION} \
    && ./configure --prefix=${PYVVO} 'CFLAGS=-w' 'CXXFLAGS=-w'\
    && make \
    && make install \
# CZMQ
    && cd "${TEMP_DIR}" \
    && wget https://archive.org/download/zeromq_czmq_${CZMQ_VERSION}/czmq-${CZMQ_VERSION}.tar.gz \
    && tar -xzf czmq-${CZMQ_VERSION}.tar.gz \
    && cd czmq-${CZMQ_VERSION} \
    && ./configure --prefix=${PYVVO} --with-libzmq=${PYVVO} 'CFLAGS=-w' 'CXXFLAGS=-w' \
    && make \
    && make install \
# FNCS
    && cd "${TEMP_DIR}" \
    && git clone -b develop --single-branch https://github.com/GRIDAPPSD/fncs.git \
    && cd fncs \
    && ./configure --prefix=${PYVVO} --with-zmq=${PYVVO} 'CFLAGS=-w' 'CXXFLAGS=-w' \
    && make \
    && make install \
    && cd python \
    && python3 setup.py sdist \
    && python3 -m pip install dist/fncs-2.0.1.tar.gz \
# GridLAB-D (connected to both MySQL and FNCS) and Xerces
    && cd "${TEMP_DIR}" \
    && git clone https://github.com/gridlab-d/gridlab-d.git -b develop --single-branch \
# Xerces
    && cd gridlab-d/third_party \
    && tar -xzf xerces-c-${XERCES_VERSION}.tar.gz \
    && cd xerces-c-${XERCES_VERSION} \
    && ./configure 'CFLAGS=-w' 'CXXFLAGS=-w' \
    && make \
    && make install \
# GridLAB-D
    && cd "${TEMP_DIR}/gridlab-d" \
    && autoreconf -isf \
    && ./configure --prefix=${PYVVO} --with-fncs=${PYVVO} --with-mysql=/usr/local/mysql --enable-silent-rules 'CFLAGS=-g -O0 -w' 'CXXFLAGS=-g -O0 -w' 'LDFLAGS=-g -O0 -w' \
    && make \
    && make install \
# Clean up source installs
    && cd "${PYVVO}" \
    && /bin/rm -rf "${TEMP_DIR}" \
# Clone pyvvo
    && git clone --recurse-submodules https://github.com/GRIDAPPSD/gridappsd-pyvvo.git \
    && cd gridappsd-pyvvo \
    && git lfs install \
    && git lfs pull \
# Install necessary python packages
    && cd pyvvo \
    && python3.7 -m pip install -r requirements.txt \
# Remove software used for building.
    && apt-get purge -y --auto-remove ${PACKAGES} \
    && apt-get -y clean

# ----------------------------------------------------
# Start bash.
# ----------------------------------------------------
ENTRYPOINT ["/bin/bash"]
# CMD ["-c", "echo 'Hello from the pyvvo container!'"]
