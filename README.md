## pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search, intended for use in CHORD.

### Installation

1. Make sure you have cuda, cupy, curand, pybind11, yaml-cpp installed.
On a CHIME/CHORD machine, this conda environment works for me:
```
    conda create -c conda-forge -n ENVNAME \
         cupy scipy matplotlib pybind11 yaml-cpp argcomplete
```
Note: I recommend the `miniforge` fork of conda, not the original conda.

2. Install the `ksgpu` library (See instructions at https://github.com/kmsmith137/ksgpu).
(This library was previously named `gputils`, but I renamed it since that name was taken on pypi.)

**Warning.** If you're building `pirate`, then you need the `chord` branch of `ksgpu`,
not the `main` branch. I'm currently trapped in Branch Divergence Hell, and the chord
branch is many commits ahead of the main branch. I hope to emerge from Branch Divergence
Hell soon!

3. Install `pirate`. The build system supports either python builds with `pip`,
or C++ builds with `make`. Here's what I recommend:
```
    # Step 1. Clone the repo and build with 'make', so that you can read
    # the error messages if anything goes wrong. (pip either generates too
    # little output or too much output, depending on whether you use -v).

    git clone https://github.com/kmsmith137/pirate
    cd pirate
    make -j 32

    # Step 2: Run some unit tests, just to check that it worked.

    python -m pirate_frb test -n 1

    # Step 3 (optional): If everything looks good, build an editable pip install.
    # This only needs to be done once per conda env (or virtualenv).
    # You can skip this step if you're content to import 'pirate'
    # directly from the build directory (this is what I do!)
    
    pip install pipmake
    pip install --no-build-isolation -v -e .    # -e for "editable" install

    # Step 4: In the future, if you want to rebuild pirate (e.g. after a
    # git pull), you can ignore pip and build with 'make'. (This is only
    # true for editable installs -- for a non-editable install you need
    # to do 'pip install' again.)

    git pull
    make -j 32   # no pip install needed, if existing install is editable

    # Step 5 (extremely optional): If you'd like autocomplete to work for the
    # 'pirate_frb' shell command, do one of these:

       # Option 1: modifies global .bashrc
       echo '"$(register-python-argcomplete pirate_frb)"' > ~/.bashrc
    
       # Option 2: modifies activate hook in current conda env 
       mkdir -p $CONDA_PREFIX/etc/conda/activate.d
       echo 'eval "$(register-python-argcomplete pirate_frb)"' \
          > $CONDA_PREFIX/etc/conda/activate.d/pirate_frb_complete.sh
```

4. For an overview for developers, see `.cursor/rules.md`.
If you're contributing to the server code (hi Dustin), some initial thoughts
are in `notes/server.md`.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
