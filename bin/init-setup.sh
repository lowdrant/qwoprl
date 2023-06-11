#!/usr/bin/env bash
#
# Setup repo
#
reporoot="$(git rev-parse --show-toplevel)"

# User-Configurable Constants
REQ_PROGRAMS=( pip3 )  # array of commands
INSTALLER='apt install'  # installer command WITHOUT sudo
ENVDIR="$reporoot"/env  # envdir
REQTXT="$reporoot"/bin/requirements.txt  # pip3 requirements file

# Install Check
for p in "${REQ_PROGRAMS[@]}"; do
    if ! which $p &> /dev/null; then
        echo "Could not find: $p. Attempting install..."
        sudo $INSTALLER $p
    else
        echo "$p already installed! Skipping"
    fi
done

# Setup VEnv
echo "Configuring virtualenv at $ENVDIR"
python3 -m venv "$ENVDIR"
pushd "$reporoot" &> /dev/null
    source "$ENVDIR"/bin/activate
    pip3 install -r "$REQTXT"
popd &> /dev/null