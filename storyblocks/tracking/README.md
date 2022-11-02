# sb object detection and tracking in pytorch serving

## local development

first, run the init script -- this will download the current version of our sb internal tracking and save it in
`model/code/tracking`

the whole game here is to create locally what sagemaker will create on deploy. That is basically two things:

1. a container that is framework- and version-specific
2. a directory of code and files that will be loaded into that container when it is run

to get the container, build it from one of the dockerfiles in this repo (files like
`pytorch/inference/docker/X.Y.Z/py3/Dockerfile.cpu`). I set up a pycharm docker run config to build these, but they
basically just amount to:

```shell script
docker build -f Dockerfile.cpu -t pytorch-inference:X.Y.Z-cpu-py36 .
```

to iterate on the second goal (a directory of code that is mounted into the container), we will run that container we
built and directly mount the files that we will eventually deploy as `model.tar.gz` at `/opt/ml/model` within our local
container. this mount *must* be `ro`, as the sagemaker deployment mounts it into a `ro` directory.

I also did this running and mounting via a pycharm run config, which amounts to:

```shell script
docker build -f Dockerfile.cpu -t sb-tracking:1.12.0-cpu-py3 .
docker run \
    -p 8080:8080 -p 8081:8081 \
    --env SAGEMAKER_PROGRAM=inference.py \
    --env AWS_ACCESS_KEY_ID=[fill this in] \
    --env AWS_SECRET_ACCESS_KEY=[fill this in] \
    --name sb-tracking \
    --rm \
    sb-tracking:1.12.0-cpu-py3 serve
```
