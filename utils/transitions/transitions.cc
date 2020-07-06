#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cuddObj.hh"

#include "SymbolicSet.hh"
#include "SymbolicModelGrowthBound.hh"

#include "TicToc.hh"
#include "RungeKutta4.hh"
#include "FixedPoint.hh"

using namespace std;
using namespace scots;


// Struct to hold the dynamics
struct Transition {
    vector<double> post;
    vector<double> r;

    Transition() { 
        post = vector<double>();
        r = vector<double>();
    }

    Transition(vector<double> post, vector<double> r) : post(post), r(r) { }
};


//template<class stateType, class inputType>
class SymbolicModelTransitions : public SymbolicModel {
public:
    using SymbolicModel::SymbolicModel;

    // This function generates the transition BDD based on the transitions as given by COSYNNC
    // It is a carbon copy of the original function "computeTransitionRelation" with some adaptions for the loaded transitions
    void loadTransitionRelation(vector<Transition>** transitions) {
        /* create the BDD's with numbers 0,1,2,.., #gridPoints */
        size_t dim = stateSpace_->getDimension();
        const size_t* nvars = stateSpace_->getNofBddVars();
        BDD **bddVars = new BDD*[dim];
        for(size_t n = 0, i = 0; i < dim; i++) {
            bddVars[i] = new BDD[nvars[i]];
            for(size_t j = 0; j < nvars[i]; j++)  {
                bddVars[i][j] = ddmgr_->bddVar(postVars_[n+j]);
            }
            n += nvars[i];
        }

        const size_t* ngp = stateSpace_->getNofGridPoints();
        BDD **num = new BDD*[dim];
        for(size_t i = 0; i<dim; i++) {
            num[i] = new BDD[ngp[i]];
            int *phase = new int[nvars[i]];
            for(size_t j=0;j<nvars[i];j++) phase[j]=0;
            for(size_t j=0;j<ngp[i];j++) {
                int *p = phase;
                int x = j;
                for (; x; x /= 2) *(p++)=0+x%2;

                num[i][j] = ddmgr_->bddComputeCube(bddVars[i],(int*)phase,nvars[i]);
            }
            delete[] phase;
            delete[] bddVars[i];
        }
        delete[] bddVars;

        /* bdd nodes in pre and input variables */
        DdManager *mgr = ddmgr_->getManager();
        size_t ndom = nssVars_ + nisVars_;
        int*  phase = new int[ndom];
        DdNode** dvars = new DdNode*[ndom];
        for(size_t i = 0; i < nssVars_; i++)
            dvars[i] = Cudd_bddIthVar(mgr, preVars_[i]);
        for(size_t i = 0; i < nisVars_; i++)
            dvars[nssVars_ + i]= Cudd_bddIthVar(mgr,inpVars_[i]);

        /* initialize cell radius
        * used to compute the growth bound */
        vector<double> eta(dim);
        stateSpace_->copyEta(&eta[0]);

        vector<double> z(dim);
        stateSpace_->copyZ(&z[0]);

        vector<double> r(dim);

        vector<double> first(dim);
        stateSpace_->copyFirstGridPoint(&first[0]);

        vector<double> inputEta(inputSpace_->getDimension());
        inputSpace_->copyEta(&inputEta[0]);

        vector<double> inputFirst(inputSpace_->getDimension());
        inputSpace_->copyFirstGridPoint(&inputFirst[0]);

        transitionRelation_ = ddmgr_->bddZero();
        const int* minterm;

        /* compute constraint set against the post is checked */
        size_t n=ddmgr_->ReadSize();
        int* permute = new int[n];
        for(size_t i=0; i<nssVars_; i++)
        permute[preVars_[i]]=postVars_[i];
        BDD ss = stateSpace_->getSymbolicSet();
        BDD constraints=ss.Permute(permute);
        delete[] permute;

        /** big loop over all state elements and input elements **/
        for(begin(); !done(); next()) {
            progress();
            minterm=currentMinterm();

            /* current state */
            vector<double> x(dim);
            stateSpace_->mintermToElement(minterm, &x[0]);
            /* current input */
            vector<double> u(inputSpace_->getDimension());
            inputSpace_->mintermToElement(minterm, &u[0]);

            /* cell radius (including measurement errors) */
            //for(size_t i=0; i<dim; i++)
            //  r[i]=eta[i]/2.0+z[i];


            // Calculate state index
            unsigned long stateIndex = 0;
            for(size_t dim = 0; dim < stateSpace_->getDimension(); dim++) {
                long indexOnAxis = round((x[dim] - first[dim]) / eta[dim]);

                if(dim == 0) stateIndex = indexOnAxis;
                else stateIndex += indexOnAxis * stateSpace_->getNofGridPoints()[dim - 1]; 
            }

            unsigned long inputIndex = 0;
            for(size_t dim = 0; dim < inputSpace_->getDimension(); dim++) {
                long indexOnAxis = round((u[dim] - inputFirst[dim]) / inputEta[dim]);

                if(dim == 0) inputIndex = indexOnAxis;
                else inputIndex += indexOnAxis * inputSpace_->getNofGridPoints()[dim - 1]; 
            }

            // Load the transition from the transitions
            auto ends = transitions[stateIndex][inputIndex];
            if(ends.size() > 0) {
                for(auto end : ends) {
                    // Find post and r from transitions
                    for(size_t dim = 0; dim < stateSpace_->getDimension(); dim++) {
                        x[dim] = end.post[dim];
                        r[dim] = end.r[dim];
                    }

                    /* determine the cells which intersect with the attainable set*/
                    /* start with the computation of the indices */
                    BDD post = ddmgr_->bddOne();
                    for(size_t i = 0; i < dim; i++) { 
                        int lb = std::lround(((x[i] - r[i] - z[i] - first[i]) / eta[i]));
                        int ub = std::lround(((x[i] + r[i] + z[i] - first[i]) / eta[i]));
                        if(lb < 0 || ub >= (int)ngp[i]) {
                            post = ddmgr_->bddZero();
                            break;
                        }

                        // WARD: This is filling between lower and upper vertices of post hyper rectangle
                        BDD zz = ddmgr_->bddZero();
                        for(int j = lb; j <= ub; j++) {
                            zz |= num[i][j]; // WARD: num[i][j] is a BDD representation of the index
                        }
                        post &= zz;
                    }

                    if(!(post == ddmgr_->bddZero()) && post <= constraints) {
                        /* compute bdd for the current x and u element and add x' */
                        for(size_t i = 0; i < nssVars_; i++)
                            phase[i] = minterm[preVars_[i]];
                        for(size_t i = 0; i < nisVars_; i++)
                            phase[nssVars_+i] = minterm[inpVars_[i]];

                        BDD current(*ddmgr_,Cudd_bddComputeCube(mgr,dvars,phase,ndom));
                        
                        current &= post;
                        transitionRelation_ +=current;
                    }
                }
            }
        }

        for(size_t i=0; i<dim; i++) 
            delete[] num[i];

        delete[] num;
        delete[] dvars;
        delete[] phase;
    }
};


// Splices the string into a vector of strings for further processing
vector<string> spliceString(string str, string denominator) {
    vector<string> splicedString;

    auto position = str.find(denominator);
    while(position != string::npos) {
        splicedString.push_back(str.substr(0, position));
        str = str.substr(position + 1, str.length());

        position = str.find(denominator);
    }

    return splicedString;
}


// Processes the information presented in a space string into a symbolic set
SymbolicSet processSpaceString(Cudd& cudd, string spaceString) {
    auto splicedString = spliceString(spaceString, " ");

    unsigned int dimension = std::stoi(splicedString[0]);

    double lowerBound[dimension];
    double upperBound[dimension];
    double eta[dimension];

    for(unsigned int i = 0; i < dimension; i++) {
        lowerBound[i] = std::stod(splicedString[1 + i]);
        upperBound[i] = std::stod(splicedString[(dimension + 1) + i]);
        eta[i] = std::stod(splicedString[(2 * dimension + 1) + i]);
    }

    SymbolicSet set(cudd, dimension, lowerBound, upperBound, eta);
    set.addGridPoints();

    return set;
}


// Loads the space information from a given filename and processes it into a space set
SymbolicSet loadSpaceFromFile(Cudd& cudd, string filename, unsigned int start, unsigned int end) {
    ifstream file(filename, std::ios_base::in);

    SymbolicSet* setPointer = nullptr;

    if(file.is_open()) {
        string line;
        unsigned long lineNumber = 0;

        string spaceString = "";

        while(getline(file, line)) {
            line = line.substr(0, line.length() - 1);

            if(lineNumber >= start && lineNumber <= end) {
                spaceString += line;
                if(lineNumber == start) spaceString += " ";
            }

            if(lineNumber == end) {
                setPointer = new SymbolicSet(processSpaceString(cudd, spaceString));
                break;
            } 

            lineNumber++;
        }
    }

    file.close();

    SymbolicSet set(*setPointer);
    set.addGridPoints();

    delete setPointer;

    return set;
}


// Returns the cardinality of the space
unsigned long getSpaceCardinality(SymbolicSet& space) {
    auto gridPoints = space.getNofGridPoints();

    unsigned long cardinality = 0;
    for(size_t dim = 0; dim < space.getDimension(); dim++) {
        if(dim == 0) cardinality = gridPoints[dim];
        else cardinality *= gridPoints[dim];
    }

    return cardinality;
}


// Loads the transitions of the plant
vector<Transition>** loadTransitionsFromFile(string filename, SymbolicSet& stateSpace, SymbolicSet& inputSpace) {
    auto stateSpaceDimension = stateSpace.getDimension();
    auto inputSpaceDimension = inputSpace.getDimension();

    auto stateSpaceCardinality = getSpaceCardinality(stateSpace);
    auto inputSpaceCardinality = getSpaceCardinality(inputSpace);

    // Initialize transitions array
    vector<Transition>** transitions = new vector<Transition>*[stateSpaceCardinality];
    for(unsigned long i = 0; i < stateSpaceCardinality; i++) {
        transitions[i] = new vector<Transition>[inputSpaceCardinality];
        for(unsigned long j = 0; j < inputSpaceCardinality; j++) {
            transitions[i][j] = vector<Transition>();
        }
    }

    // Load transitions from file
    ifstream file(filename, std::ios_base::in);

    if(file.is_open()) {
        string line;
        unsigned long lineNumber = 0;

         while(getline(file, line)) {
            line = line.substr(0, line.length() - 1);

            if(lineNumber > 13 && line != "") {
                auto content = spliceString(line, " ");

                // Find state index
                unsigned long stateIndex = 0;
                for(size_t dim = 0; dim < stateSpaceDimension; dim++) {
                    if(dim == 0) stateIndex = stol(content[dim]);
                    else stateIndex += stol(content[dim]) * stateSpace.getNofGridPoints()[dim - 1];                   
                }

                // Find input index
                unsigned long inputIndex = 0;
                for(size_t dim = 0; dim < inputSpaceDimension; dim++) {
                    if(dim == 0) inputIndex = stol(content[dim + stateSpaceDimension]);
                    else inputIndex += stol(content[dim + stateSpaceDimension]) * inputSpace.getNofGridPoints()[dim - 1];
                }

                // Format post
                vector<double> post;
                for(size_t dim = 0; dim < stateSpaceDimension; dim++) {
                    post.push_back(stod(content[dim + stateSpaceDimension + inputSpaceDimension]));
                }

                // Format r
                vector<double> r;
                for(size_t dim = 0; dim < stateSpaceDimension; dim++) {
                    r.push_back(stod(content[dim + 2 * stateSpaceDimension + inputSpaceDimension]));
                }

                // Set transition
                transitions[stateIndex][inputIndex].push_back(Transition(post, r));
            }

            lineNumber++;
        }
    }
    file.close();

    return transitions;
}


// Main function to run the program
int main() {
    TicToc tt;
    Cudd cudd;

    string filename = "plant.trs";
    size_t specification = 1; // 0: invariance, 1: reachability, 2: reach and stay

    // Load spaces from abstraction file
    auto stateSpace = loadSpaceFromFile(cudd, filename, 2, 5);
    auto stateSpaceDimension = stateSpace.getDimension();
    auto stateSpaceCardinality = getSpaceCardinality(stateSpace);
    std::cout << "State space dimension: " << stateSpaceDimension << " - State space cardinality: " << stateSpaceCardinality << std::endl;

    auto inputSpace = loadSpaceFromFile(cudd, filename, 7, 10);
    auto inputSpaceDimension = inputSpace.getDimension();
    auto inputSpaceCardinality = getSpaceCardinality(inputSpace);
    std::cout << "Input space dimension: " << inputSpaceDimension << " - Input space cardinality: " << inputSpaceCardinality << std::endl << std::endl;

    SymbolicSet stateSpacePost(stateSpace, 1);

    // Load the transitions of the plant
    tt.tic();
    auto transitions = loadTransitionsFromFile(filename, stateSpace, inputSpace);
    tt.toc();
    std::cout << "Transitions loaded from file: " << filename << std::endl << std::endl;

    // Define the abstraction using the loaded transitions
    tt.tic();
    SymbolicModelTransitions abstraction(&stateSpace, &inputSpace, &stateSpacePost);
    abstraction.loadTransitionRelation(transitions);
    tt.toc();
    std::cout << "Transitions processed into a BDD" << std::endl << std::endl;
    
    // Define target set
    SymbolicSet target(stateSpace);
    double H[4 * stateSpaceDimension]={-1, 0,
                     1, 0,
                     0, -1,
                     0, 1};
    double h[4] = {1.0, 1.0, 1.0, 1.0};
    target.addPolytope(4, H, h, INNER);

    // Fixed point iteration to calculate controller
    std::cout << "Abstraction generated, continuing with synthesizing controller" << std::endl;

    FixedPoint fp(&abstraction);
    BDD targetBdd = target.getSymbolicSet();

    BDD controllerBdd = cudd.bddZero();

    switch(specification) {
    	case 0:
    		tt.tic();
		    controllerBdd = fp.safe(targetBdd, 1);
		    tt.toc();
    		break;
    	case 1:
    		tt.tic();
		    controllerBdd = fp.reach(targetBdd, 1);
		    tt.toc();
    		break;
    	case 2:
    		tt.tic();
			size_t i,j;
			BDD X = cudd.bddOne(); BDD XX = cudd.bddZero();
			BDD Y = cudd.bddZero(); BDD YY = cudd.bddOne();

			BDD U = inputSpace.getCube();
			for(i = 1; XX != X; i++) {
				X = XX;
				BDD preX = fp.pre(X);
				YY = cudd.bddOne();
				for(j=1; YY != Y; j++) {
					Y = YY;
					YY = ( fp.pre(Y) & targetBdd ) | preX;
				}
				XX=YY; 
				std::cout << "Iterations inner: " << j << std::endl;
				BDD N = XX & (!(controllerBdd.ExistAbstract(U)));
				controllerBdd = controllerBdd | N;
			}
			std::cout << "Iterations outer: " << i << std::endl;
			tt.toc();
    		break;
    }

    // Save controller
    SymbolicSet controller(stateSpace, inputSpace); // This controller symbolic set is the cartesian product of the two sets (X * U)
    controller.setSymbolicSet(controllerBdd);

    std::cout << "Domain size: " << controller.getSize() << std::endl;
    controller.writeToFile("transitions_controller.bdd");

    // Get winning domain
    auto winningBdd = controllerBdd;
    SymbolicSet winningDomain(stateSpace, 0);
    winningDomain.setSymbolicSet(winningBdd);
    auto winningDomainSize = winningDomain.getSize();
    std::cout << std::endl << "Winning domain size: " << winningDomainSize << " - " << (double)winningDomainSize/(double)stateSpaceCardinality*100 << "%" << std::endl;

    // Free memory
    for(unsigned long i = 0; i < stateSpaceCardinality; i++) 
        delete[] transitions[i];
    delete[] transitions;

    return 1;
}

