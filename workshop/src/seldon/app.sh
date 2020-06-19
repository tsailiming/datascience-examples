#!/bin/sh

MODEL_NAME=LRModel
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0

echo "starting microservice"
exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
