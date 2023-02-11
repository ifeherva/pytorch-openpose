#!/bin/bash

# start server
torchserve --model-store ./torchserve/ --models openpose18.mar --stop --ts-config torchserve/config.properties

# probe with an image
curl http://localhost:8080/predictions/openpose18 -T "$1"