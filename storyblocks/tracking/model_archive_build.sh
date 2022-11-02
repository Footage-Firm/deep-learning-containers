#!/usr/bin/env bash

HERE=$(pwd)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=${DIR}/model
CODE_DIR=${MODEL_DIR}/code
TRACKING_DIR="${CODE_DIR}/tracking"

# make sure that the yaefd_curren_sha file is up to date
cd "${TRACKING_DIR}" || exit
TRACKING_CURRENT_SHA=$(git rev-parse HEAD)
cd "${CODE_DIR}" || exit
echo "${TRACKING_CURRENT_SHA}" > tracking_current_sha

# add a timestamp while we are here
date -u +"%Y%m%dT%H%M%S" > model_archive_timestamp

# create the archive
cd "${MODEL_DIR}" || exit
if [ -f "model.tar.gz" ] ; then
    rm "model.tar.gz"
fi
tar --exclude "*.git" --exclude="*.png" --exclude="*.jpg" -czvf model.tar.gz ./*
mv model.tar.gz ../

cd "${HERE}" || exit

exit $?
