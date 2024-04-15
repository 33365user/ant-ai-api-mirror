#!/bin/bash

echo "Checking System Python Version..."

PYVER=$(python3 --version)
PYPATCHVER=$(echo $PYVER | awk -F' ' '{print $2}' | awk -F'.' '{print $3}')
PYMINVER=$(echo $PYVER | awk -F' ' '{print $2}' | awk -F'.' '{print $2}')
PYMAJVER=$(echo $PYVER | awk -F' ' '{print $2}' | awk -F'.' '{print $1}')

if [ $PYMAJVER -ne 3 ]; then
    echo "Python Major Version Not Valid (Python V3 Required). Version Present: '$PYMAJVER'"
    exit
fi

if [ $PYMINVER -lt 8 ] || [ $PYMINVER -gt 12 ]; then
    echo "Python Minor Version Not Valid. Version Present: $PYMINVER. Expected 8 <= '$PYMINVER' <= 12"
    exit
fi

echo "Python Version: $PYMAJVER.$PYMINVER.$PYPATCHVER"

echo "Creating Environment for venvs..."
echo "Default Directory: '~/venvs'"

mkdir -p $HOME/venvs
cd $HOME/venvs/
echo "#------Creating Backend API Venv-----#"
mkdir -p $HOME/venvs/backendvenv
# This one uses 3.11.6 as the primary version
echo "Downloading from Apt..."
sudo apt-get install python3.11-venv
echo "Downloading Complete!"

echo "Current Directory: "$PWD
echo "Making Virtual Environment..."
python3.11 -m venv backendvenv
echo "Activating Virtual Environment...."
source $HOME/venvs/backendvenv/bin/activate
echo "Installing Pip Modules..."
python3.11 -m pip install -r $HOME/scripts/backend_requirements.txt
deactivate
echo "Pip Modules Installed Successfully!"
echo "#-----Backend API Venv Complete-----#"
echo "#----- ----- ----- ----- ----- -----#"

cd ~/venvs
echo "#----- Creating AntAI Venv     -----#"
mkdir -p $HOME/venvs/antaivenv
# This one uses 3.10 as the primary version
echo "Downloading from Apt..."
sudo apt-get install python3.10-venv
sudo apt-get install build-essential
echo "Downloading Complete!"

echo "Current Directory: "$PWD
echo "Making Virtual Environment..."
python3.10 -m venv antaivenv
echo "Activating Virtual Environment...."
source $HOME/venvs/antaivenv/bin/activate
echo "Installing Pip Modules..."
python3.10 -m pip install wheel
python3.10 -m pip install -r $HOME/scripts/antai_requirements.txt
echo "Installing IPython (Separately)"
python3.10 -m pip install IPython[all]
deactivate
echo "Pip Modules Installed Successfully!"
echo "#----- AntAI API Venv Complete -----#"
echo "#----- ----- ----- ----- ----- -----#"