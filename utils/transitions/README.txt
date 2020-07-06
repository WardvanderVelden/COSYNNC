Utility tool for loading COSYNNC transitions and generating SCOTS controllers based upon those.

In order to use the tool:
* Properly configure the Makefile according to where SCOTS and the CUDD library are installed.
* In the main function, configure the variable "filename" such that it points to the COSYNNC generated .trs file that contains the transitions.
* Run make
* Run ./transitions

