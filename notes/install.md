# Installation

**NOTE.** Currently, this code can only be compiled on a recent ubuntu
linux machine with **a physical GPU**, and the cuda toolkit installed.
I hope to improve this in the future!

1. Set up a conda environment. `pirate` relies on the **system** CUDA toolkit
(`nvcc`, CUDA headers) and the **system** host compiler (`gcc`/`g++`); everything
else is conda-installed. The repo ships two environment files:

   - `environment_minimal.yml` -- just enough to build and run `pirate`.
   - `environment_dev.yml` -- a superset of the minimal env that also installs
     scipy/matplotlib, jupyterlab, the sphinx docs toolchain, and some
     editor/shell tools. Use this if you're doing development.

To create the env:
```
    conda env create -n ENVNAME -f pirate/environment_dev.yml   # or pirate/environment_minimal.yml
    conda activate ENVNAME
```
I recommend the `miniforge` fork of conda, not the original conda.

2. Install the `ksgpu` library (see instructions at https://github.com/kmsmith137/ksgpu).

3. Install `pirate`. The build system supports either python builds with `pip`,
or C++ builds with `make`. Here's what I recommend:
```
    # Step 1. Clone the repo and build with 'make', so that you can read
    # the error messages if anything goes wrong. (pip either generates too
    # little output or too much output, depending on whether you use -v).

    git clone --recursive https://github.com/kmsmith137/pirate
    cd pirate
    make -j 32

    # Step 2: Run some unit tests, just to check that it worked.

    python -m pirate_frb test -n 1

    # Step 3 (optional): If everything looks good, build an editable pip install.
    # This will let you import 'pirate' outside the build dir, and run with the
    # syntax 'pirate_frb CMD ARGS' instead of 'python -m pirate_frb CMD ARGS'
    # This only needs to be done once per conda env (or virtualenv).
    
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
       echo 'eval "$(register-python-argcomplete pirate_frb)"' >> ~/.bashrc
    
       # Option 2: modifies activate hook in current conda env 
       mkdir -p $CONDA_PREFIX/etc/conda/activate.d
       echo 'eval "$(register-python-argcomplete pirate_frb)"' \
          > $CONDA_PREFIX/etc/conda/activate.d/pirate_frb_complete.sh
```
