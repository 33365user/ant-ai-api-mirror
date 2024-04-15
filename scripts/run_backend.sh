echo "Attempting to activate Backend Scripts..."
echo "Script is non-functional - exiting."
exit 1

# Function to run commands in the background
run_in_background() {
    echo "Executing Background Task..."
    "$@"
    local pid=$!
    echo $pid
}

if [ -e "$HOME/venvs/backendvenv/bin/activate" ]; then
    echo "Running Backend API..."

    # Start the first set of commands in the background
    pid1=$($HOME/venvs/backendvenv/bin/python3.11 "-m flask run" &)
    pid2used=1

    read -e -p "Would you like to deploy the AntAI Model (Currently not available) - If not, the built-in substitute RPC Server will be used (Y/n)? " choice

    if [[ "$choice" == [Yy]* ]]; then
        echo "ERROR Not Implemented!"
        pid2used=0
    else
        pid2=$($HOME/venvs/backendvenv/bin/python3.11 $HOME/backend/test.py &)
    fi

    # Trap SIGINT (Ctrl+C) to exit both background processes
    echo "Trapping SIGINT..."
    if [ "$pid2used" -eq 0 ]; then
        trap 'kill $pid1; exit' INT
    else
        trap 'kill $pid1; kill $pid2; exit' INT
    fi

    # Frontend code or user interface here
    echo "Press Ctrl+C to exit ..."

    # Wait for processes to finish
    wait $pid1

    if [ "$pid2used" -eq 1 ]; then
        wait $pid2
    fi

    echo "Processes Exited."

else
    echo "The venv doesn't exist. Cancelling"
    exit 1
fi