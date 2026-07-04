# Installation

**NOTE.** Currently, this code can only be compiled on a recent ubuntu
linux machine with **a physical GPU**, and the cuda toolkit installed.
I hope to improve this in the future!

1. Set up a conda environment. `pirate` relies on the **system** CUDA toolkit
(`nvcc`, CUDA headers) and the **system** host compiler (`gcc`/`g++`); everything
else is conda-installed. The repo ships two environment files -- pick one:

   - `environment_minimal.yml` -- just enough to build and run `pirate`.
   - `environment_dev.yml` -- a superset of the minimal env that also installs
     scipy/matplotlib, jupyterlab, the sphinx docs toolchain, and some
     editor/shell tools. Use this if you're doing development.

   Create the env from the parent directory that contains the `ksgpu/` and
   `pirate/` checkouts:
```
    conda env create -n ENVNAME -f pirate/environment_dev.yml   # or pirate/environment_minimal.yml
    conda activate ENVNAME
```
   Each yml file's header comment documents its package pins (and, for the dev
   env, how to build the LaTeX-based docs). I recommend the `miniforge` fork of
   conda, not the original conda.

2. Check out the `asdf-cxx` git submodule (used by pirate's C++ code), install
the `pipmake` build backend from PyPI, then do editable `pip install`s of `ksgpu`
and `pirate`.
```
    (cd pirate && git submodule update --init --recursive)
    pip install pipmake
    (cd ksgpu  && pip install --no-build-isolation -v -e .)
    (cd pirate && pip install --no-build-isolation -v -e .)
```
Here, `--no-build-isolation` is needed since we assumed that `ksgpu` is a sibling
checkout (rather than pulled from PyPI).
Since `--no-build-isolation` does not auto-install build deps, we installed `pipmake`
explicitly first.

3. Verify the build by running some unit tests:
```
    python -m pirate_frb test -n 1
```
   After the editable install above, you can also run this as `pirate_frb test -n 1`.

**Tip (debugging a broken build).** You can build `ksgpu` or `pirate` with
`make -j 32` directly in the checkout. `make` prints readable compiler errors,
whereas pip generates either too little or too much output depending on whether
you pass `-v`. Because the installs above are editable, a later `make -j 32`
(e.g. after `git pull`) is picked up automatically, with no need to re-run
`pip install`.

4. (Optional) Enable tab-completion for the `pirate_frb` shell command:
```
       # Option 1: modifies global .bashrc
       echo 'eval "$(register-python-argcomplete pirate_frb)"' >> ~/.bashrc

       # Option 2: modifies activate hook in current conda env
       mkdir -p $CONDA_PREFIX/etc/conda/activate.d
       echo 'eval "$(register-python-argcomplete pirate_frb)"' \
          > $CONDA_PREFIX/etc/conda/activate.d/pirate_frb_complete.sh
```
