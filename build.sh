#!/bin/sh

# set build variables
USER_NAME=lab
CONTAINER_NAME=${PWD##*/}
PORT=8889
# -subj arg for openssl self-signed certificate if have to
SSLSUBJ=/C=US/ST=California/CN=DOCA
# create example env configuration to use in the container
#tee init.cnf <<EOF
#ROOT=/home/$USER_NAME
#PORT=$PORT
#CONTAINER_NAME=$CONTAINER_NAME
#JUPYTER_PASS=
#SHM_SIZE=$SHM_SIZE
#MAX_BUFFER_SIZE=10000000000
#IOPUB_DATA_RATE_LIMIT=10000000000
#EOF

# actual configuration file
#cp init.cnf env.cnf

# cleanup: remove all
docker system prune -a
# remove previous build
docker rm $CONTAINER_NAME

# build with arguments
docker build --no-cache --force-rm \
             --build-arg USER_NAME=$USER_NAME \
             --build-arg PYTHON=python3.10 \
             --build-arg TZ=America/Los_Angeles \
             -t $CONTAINER_NAME .
             
# remove intermediate images
docker image prune --filter label=stage=builder

