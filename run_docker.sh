docker run -it --rm \
    --gpus all \
    --net="host" \
    --shm-size 16g \
    -e NVIDIA_VISIBLE_DEVICES=... \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/${USER}:/home/${USER}\
    -v /media/${USER}:/media/${USER} \
    --name ${USER}.tong3 \
    mmdet-pytorch1.9-cuda111:PVTmmcv1.6

#mmdet-pytorch1.9-cuda111:latest
