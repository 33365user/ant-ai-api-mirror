#!/bin/bash
echo "This script will uninstall all components previously installed by other scripts."
read -e -p "Would you like to continue this operation (Y/n)? " choice

if [[ "$choice" == [Yy]* ]]; then
  echo "Continuing..."
else
  exit 1
fi

# Will remove the directories involved with everything, but scripts.
echo "Do note that the scripts folder will not be removed (as the host of this file)"

rm -Rf $HOME/backend
rm -Rf $HOME/venvs

echo "This script is about to remove $HOME/Ant_Project_TF2"
echo "Would you like to remove this folder (Y/n)?"
if [[ "$choice" == [Yy]* ]]; then
  echo "Removing $HOME/Ant_Project_TF2"
  rm -Rf $HOME/Ant_Project_TF2
  echo "Script Complete!"
else
  echo "Script Complete!"
  exit
fi