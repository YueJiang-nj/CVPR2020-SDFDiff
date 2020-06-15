#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv sdfdiff
echo "Activating virtual environment"

source $PWD/sdfdiff/bin/activate

$PWD/sdfdiff/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
