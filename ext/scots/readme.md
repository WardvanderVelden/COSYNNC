# SCOTS: Automated Synthesis of Symbolic Controllers for General Non-lineal Systems

> SCOTSv0.2 is in the making. If you are feeling adventurous have
> look at [SCOTSv0.2](https://gitlab.lrz.de/matthias/SCOTSv0.2)

**SCOTS** is a C++ tool (with a small Matlab interface) to synthesize controller for
possibly perturbed nonlinear control systems with respect to safety and reachability specifications.

Please read the manual in the manual directory ./manual/manual.pdf

## Requirements

SCOTS requires only a modern C++ development environment where you can compile C++ (v11) source codes.
All other requirements are included with SCOTS.

To make full use of SCOTS, you may need to have MATLAB installed to be able to simulate the synthesized controllers using the provided MATLAB interface in [/mfiles](/mfiles).

SCOTS was originally developed to work in Linux and MacOS. However, we managed to make it work in Windows and included a small help to guide you [here](/installation_notes_windows.txt). SCOTS is known to work much slower in Windows.

## Installation

SCOTS itself is a header only library. You only need to add SCOTS source directory to the include directory in the compiler command when you work with SCOTS. However, SCOTS depends on the CUDD library to represent the data structures used in SCOTS.
CUDD library is inlcuded in the directory [/cudd-3.0.0](/cudd-3.0.0).

The CUDD library by is developed Fabio Somenzi and we include it in this repo since it is no longer publicly available. SCOTS uses also the dddmp and C++ wraper of the CUDD library (also included). Follow the follwoing steps to compile and install the CUDD library.

- Navigate to the directory and configured the library:  

    `$ ./configure --enable-shared --enable-obj --enable-dddmp --prefix=/opt/local/`

- Now, make the libtrary and install it:

    `$ make`

    `$ sudo make install`

- We noticed that, on some linux systems, the files **config.h** and **util/util.h** do not get copied after  the installation. You have to copy them manually to **/opt/local/include/** if they are not copied.

- Finally, add the path to the installed library to your Linux's library search path by the running command for each terminal session:

    `$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/local/lib`

Further details are found in the readme files in example directories and in the [manual](/manual/manual.pdf).

For installing and running SCOTS on Windows, please refer to the [Windows installation help file](/installation_notes_windows.txt).

## Directory structure

- **./bdd/**:

    Contains the source C++ source code for the SCOTS classes which use Binary Decision Diagrams as underlying data structure.

- **./doc/**:

    C++ Class documentation directory.
  
- **./examples/**:

    Some C++/Maltab programs demonstrating the usage of SCOTS.
  
- **./manual/**:

    Contains a the manuel with its tex source.
  
- **./mfiles/**:

    Contains an mfile as a wrapper to the mex-file functions.
  
- **./mfiles/mexfiles/**:

    mex-file to read the C++ output from file.

## Support

Please report any problems/bugs you face while installing and running SCOTS to [Mahmoud Khaled](http://hyconsys.com/members/mkhaled/).
