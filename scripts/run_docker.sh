#!/usr/bin/env bash
cd ..
docker build . -t robiniowitsch/cfc-model-server
docker run -p 8000:8000 robiniowitsch/cfc-model-server:latest
cd scripts