#!/bin/bash
echo "This Script will download from Github the AntAIBackend - assuming this System has a valid Deployment Key"
echo "Creating Directory..."

mkdir -p $HOME/backend

read -p "Please Enter the name of the Repo's Owner: " owner
read -p "Please Enter the name of the Repo: " reponame

if [-z "$owner"]; then
  echo "Improper Input for Owner."
  exit 1
fi

if [-z "$reponame"]; then
  echo "Improper Input for Repo Name"
  exit 1
fi

echo "Attempting to Clone Repo to $HOME/backend..."

git clone "git@github.com:$owner/$reponame" "$HOME/backend"

echo "Making Directories..."
mkdir $HOME/backend/images
mkdir $HOME/backend/uploads

chmod 777 -R $HOME/backend