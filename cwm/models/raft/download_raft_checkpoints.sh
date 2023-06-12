#!/bin/bash

ckpt_dir=${1:-"../../../checkpoints"}
raft_dir="${ckpt_dir}/raft_checkpoints"

mkdir -p $ckpt_dir
mkdir -p $raft_dir

wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm ./models.zip
mv ./models/raft-sintel.pth "${raft_dir}/raft-large.pth"
mv ./models/raft-small.pth "${raft_dir}/raft-small.pth"
rm -r ./models

raft_ckpts=`ls ${raft_dir}`
echo "Downloaded RAFT checkpoints:"
echo "${raft_ckpts}"
