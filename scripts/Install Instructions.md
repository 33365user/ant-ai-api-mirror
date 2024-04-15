# Installation Instructions for Idantifier Backend

---

## 1.1 Setting Up
- Setup Ubuntu Linux LTS (20.04 or compatible, preferably newer) manually
- Add [dockerinstall.sh](dockerinstall.sh) (in /scripts) to the file system (via any means)
- On the File System, it is recommended to structure like so:
```
  ~/scripts
    | dockerinstall.sh
    | setup_git_backend.sh
    | setup_venvs.sh
    | antai_requirements.txt
    | backend_requirements.txt

  ~/venvs
  
  ~/backend
  
  ~/Ant_Project_TF2

```
- Modify access permissions to allow execution, then run the script. (This will work for all shell files on record)
    ```sh
      sudo chmod +x dockerinstall.sh
      ./dockerinstall.sh
    ```
- Note: the rest of this code may be performed inside this scope, or inside a ubuntu docker container - either works, depending on context.
- Copy the rest of the scripts in the /scripts folder from this repo into the /scripts folder of your machine.
- Run the following commands to update the system indexes of the apt repositories:
    ```
      sudo apt-get upgrade
      sudo apt-get update
  ```
- Run [setup_venvs.sh](setup_venvs.sh) in the local user context (not sudo) to create the venvs in the `/home/<user>/venvs` directory.
- It may ask for sudo permissions, grant it.
- Set up the files on a private GitHub repository - and go through the process of making a deployment key. 
  - See [Managing Deploy Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys#deploy-keys)
  - See [Generating an SSH Key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)
- Use [setup_git_backend.sh](setup_git_backend.sh) to download the AntAIBackend from Github to the `/home/<user>/backend` folder.
  - Make sure to refer to the proper Repo Owner and Repository Name!
- Until this is automated, the following process must be done for the flask server to run - it will consume the session, with note that the proper virtual environment must be activated, in the test case, that is the backendvenv (V. 3.11)
  ```shell
    source $HOME/venvs/backendvenv/bin/activate
    cd $HOME/backend
    waitress-serve --port=5000 app:app
  ```
  - Note that you may need to run the server two or three times initially such that the appropriate folders can be made - a known bug causes an OS Permissions Error despite laxity in said permissions. A Resolution is being investigated.
  - Also note that if this kind of OS error keeps happening, it could be entirely possible that the filesystem has the wrong access permissions. Use the following command to fix that (and go through running the server a few times for verification):
  ```shell
    chmod 777 -R $HOME/backend
  ```
- To activate the test RPC server for testing, do much the same, but instead we can combine it into one line(in another shell session):
  ```shell
    $HOME/venvs/backendvenv/bin/python3.11 $HOME/backend/RPCAntAIServer.py
  ```
- d


# Further Documentation
Do note that as a domain has not been selected at which to host this service at (DNS record), the Privacy Policy and Terms of Service's links won't work. Further, the application is currently hardcoded with an IP rather than a Domain Name - meaning it will not function currently without ammendments.

