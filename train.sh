#!/bin/bash

deepspeed \
    --hostfile randy/hostfile \
    randy/train.py "$@"
