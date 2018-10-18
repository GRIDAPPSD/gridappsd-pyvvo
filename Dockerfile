# Use base image for GridAPPS-D applications
FROM gridappsd/app-container-base

# We'll be putting everything in /pyvvo.
ENV PYVVO=/pyvvo

# Setup other environment variables:
# MSCC --> MySQL Connector/C
# All libs are going into /pyvvo/lib except MSCC
ENV MSCC_VERSION=6.1.11 \
    TEMP_DIR=/tmp/source \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd \
    CXXFLAGS=-I${PYVVO}/share/gridlabd \
    PACKAGES="autoconf automake cmake g++ gcc git libtool make wget"

# Copy and install Python requirements.
WORKDIR $PYVVO
RUN mkdir pyvvo
COPY ./pyvvo/requirements.txt pyvvo/requirements.txt
RUN pip install -r pyvvo/requirements.txt

# Create temporary directory for installations
#RUN mkdir ${TEMP_DIR}
WORKDIR ${TEMP_DIR}

# Install packages for installation
RUN perl -E "print '*' x 80" \
    && printf '\nInstalling packages...\n' \
    && apt-get update && apt-get -y install ${PACKAGES} \
    && rm -rf /var/lib/opt/lists/* \
# Install MySQL Connector/C
    && perl -E "print '*' x 80" \
    && printf '\nInstalling MySQL Connector/C...\n' \
    && wget "https://dev.mysql.com/get/Downloads/Connector-C/mysql-connector-c-${MSCC_VERSION}-src.tar.gz" \
    && tar -zxf "mysql-connector-c-${MSCC_VERSION}-src.tar.gz" \
    && cd "mysql-connector-c-${MSCC_VERSION}-src" \
    && cmake -G "Unix Makefiles" -DCMAKE_C_FLAGS="-w" -DCMAKE_CXX_FLAGS="-w"\
    && make \
    && make install \
    && ln -s /usr/local/mysql /usr/local/mysql-connector-c \
# Install GridLAB-D
    && cd $TEMP_DIR \
    && git clone https://github.com/gridlab-d/gridlab-d.git -b develop --single-branch \
    && perl -E "print '*' x 80" \
    && printf '\nInstalling Xerces...\n' \
    && cd ${TEMP_DIR}/gridlab-d/third_party \
    && tar -xzf xerces-c-3.2.0.tar.gz \
    && cd ${TEMP_DIR}/gridlab-d/third_party/xerces-c-3.2.0 \
    && ./configure \
    && make \
    && make install \
    && perl -E "print '*' x 80" \
    && printf '\nInstalling GridLAB-D...\n' \
    && cd ${TEMP_DIR}/gridlab-d \
    && autoreconf -isf \
    && ./configure --prefix=${PYVVO} --with-mysql=/usr/local/mysql --enable-silent-rules 'CFLAGS=-g -O0 -w' 'CXXFLAGS=-g -O0 -w -std=c++11' 'LDFLAGS=-g -O0 -w' \
    && make \
    && make install \
# Clean up source installs
    && perl -E "print '*' x 80" \
    && printf '\nCleaning up source installations...\n' \
    && cd "${PYVVO}" \
    && /bin/rm -rf "${TEMP_DIR}" \
# Clone pyvvo
#    && git clone --recurse-submodules https://github.com/GRIDAPPSD/gridappsd-pyvvo.git \
#    && cd gridappsd-pyvvo \
#    && git lfs install \
#    && git lfs pull \
# Install necessary python packages
#    && cd pyvvo \
#    && python3 -m pip install -r requirements.txt \
# Remove software used for building.
    && perl -E "print '*' x 80" \
    && printf '\nRemoving packages...\n' \
    && apt-get purge -y --auto-remove ${PACKAGES} \
    && apt-get -y clean

# DEVELOPMENT ONLY: Install nano so we can look at file dumps.
RUN apt-get install -y nano

# Copy application code.
WORKDIR $PYVVO
COPY pyvvo ./pyvvo

# Use symbolic link to link app configuration.
RUN ln -s ${PYVVO}/pyvvo/pyvvo.config /appconfig

# Work from the 'app' directory so that pathing "just works"
WORKDIR ${PYVVO}/pyvvo/app
