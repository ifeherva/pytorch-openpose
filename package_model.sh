#!/bin/bash

torch-model-archiver --model-name openpose18 \
                     --version 1.0 \
                     --serialized-file model/openpose18.pt \
                     --handler torchserve/openpose_handler.py \
                     --requirements-file torchserve/requirements.txt \
                     --export-path torchserve \
                     --force