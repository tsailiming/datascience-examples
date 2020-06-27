#!/bin/bash

cd /projects/datascience-examples
GIT_REV=`git rev-parse --short HEAD`
curl -H 'Content-type: application/json' --data '{"tag": "'"${GIT_REV}"'"}'  http://el-pipeline.$WORKSHOP_USER_ID-prod.svc.cluster.local:8080