Utility tool for converting COSYNNC neural network static controllers to binary decision diagrams

In order to use the tool:
* Properly configure the Makefile according to where SCOTS and the CUDD library are installed.
* Inside converter.cc, configure the constants "stateCardinality" and "inputCardinality" such that they correspond to the cardinality of the static controller.
* In the main function, configure the variable "filename" such that it points to the COSYNNC generated .scs file.
* Run make
* Run ./converter

