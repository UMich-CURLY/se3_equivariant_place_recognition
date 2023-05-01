# se3_equivariant_place_recognition Docker
This folder contains instructions on creating a docker image that includes PyTorch and other libraries needed for se3_equivariant_place_recognition.
## Install Docker [Install instruction for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
After installing docker, [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/).
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

## How to build docker
build image
```
docker build --tag umcurly/equivariantpr_docker .
```
change [folder direction in line 15](https://github.com/UMich-CURLY/se3_equivariant_place_recognition/blob/e57740400760aa8f03978b7bded6f01a1ef9fd1c/docker/build_docker_container.bash#L15) to -v "<PATH_IN_LOCAL_MACHINE>:<PATH_IN_DOCKER_CONTAINER>". For example, if I want `/home/$USER/` link to the local path `/home/$USER/this_working_path/` in the local machine, I can replace line 15 as `-v /home/$USER/this_working_path/:/home/$USER/"`. If I have some data set in another directory, I can also link it using another line `-v "/home/$USER/another_working_path/dataset_directory/:/home/$USER/data/"`.
After changin the directory, then build container
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
