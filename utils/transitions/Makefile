#
# compiler
#
CC        = g++
CXXFLAGS 		= -Wall -Wextra -std=c++11 -O3 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0


#
# scots 
#
SCOTSROOT		= ../../ext/scots
SCOTSINC		= -I$(SCOTSROOT)/bdd -I$(SCOTSROOT)/utils 

#
# cudd 
#
CUDDPATH		=  /usr/local
CUDDINC 		= -I$(CUDDPATH)/include
CUDDLIBS		= -lcudd 
CUDDLPATH   = -L$(CUDDPATH)/lib

TARGET = transitions

all: $(TARGET)

%.o:%.cc
	$(CC) -c $(CXXFLAGS) $(CUDDINC) $(SCOTSINC) $< -o $@

$(TARGET): $(TARGET).o
	$(CC) $(CXXFLAGS) -o $(TARGET) $(TARGET).o $(CUDDLPATH) $(CUDDLIBS)


clean:
	rm  ./$(TARGET)  ./$(TARGET).o
