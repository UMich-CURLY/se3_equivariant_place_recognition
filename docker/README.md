# EPN-NetVLAD Docker
This folder contains instructions on creating a docker image that includes PyTorch and other libraries needed for EPN-NetVLAD.

## How to build

build image
```
docker build --tag umcurly/e2pn_docker .
```
change folder direction in line 15, then build container
```
bash build_docker_container.bash [container_name]
```
After building the container, you will enter the docker container. To work stably in docker, we recommend running `exit` and then follow the next section for running docker.

## How to use
start docker 
```
docker start [container_name]
```
run docker
```
docker exec -it [container_name] /bin/bash
```
run docker with root access
```
docker exec -u root -it [container_name] /bin/bash
```
