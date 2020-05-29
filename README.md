# COSYNNC
COSYNNC is a correct-by-design neural network controller synthesis framework, intended to synthesis controllers that adhere to predefined control specifications. The framework supports both linear and nonlinear system dynamics and has support for reachability, invariance and reach and stay specifications although these specifications can easily be extended. The resulting controllers are neural networks that can be exported as, amongst other formats, MATLAB files for easy access and validation.

<h1> Building the framework </h1>
COSYNNC relies on the open and rich neural network library called MXNet. In order to successfully build and exploit the COSYNNC framework, a build of this library is required. Thus the first step in building the COSYNNC framework is to build the MXNet library.

<h2> Building MXNet </h2>
In order to build MXNet, the dependencies upon which MXNet relies need to be configured. MXNet is a vast library and has a lot of customizable options for different configurations (CPU or GPU based etc...). For COSYNNC, the simplest form of MXNet is used which is CPU based MXNet. The build instructions for both Ubuntu and Windows will now be provided.

<h3>Ubuntu</h3>
* Install GCC

* Install CMake 1.13 or higher. This can be done by downloading the latest version of [CMake](https://cmake.org) and extracting the files. After that, open a terminal and navigate to where the files where extracted. Then run:
	```console
	sudo apt install libssl-dev
	./bootstrap
	make -j4
	sudo make install
	```
	Note: If CMake does not show the proper version a restart might be required to fix that
* Install the prerequisite packages for MXNet:
	```console
	sudo apt-get update
	sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev 
	```
* Recursively clone MXNet from github:
	```console
	git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
	```

* Copy the linux makefile to the config file
	```console
	cp config/linx.cmake config.cmake
	```

* Edit the config file such that the following parameters are set:
	```console
	CUDA OFF
	CUDNN OFF
	BLAS OPEN
	OEPNCV OFF
	OPENMP OFF
	MKLDNN OFF
	LAPACK OFF
	USE_CPP_PACKAGE ON
	```

* Build the core shared library by running:
	```console
	rm -rf build
	mkdir build
	cd build
	cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
	cmake --build .
	```

	This procedure should result in a shared library in the build folder and the required headers in the cpp-package/include folder.

<h3>Windows</h3>

<h2> Building COSYNNC </h2>
In order to build COSYNNC, the MXNet library needs to be build first as specified above. Once that has been completed follow the following instructions:

* Clone the desired version of the COSYNNC framework from the github:
	```console
	git clone https://github.com/WardvanderVelden/COSYNNC cosynnc
	```

* Configure the CMake file CMakeList.txt by changing the parameter MXNET_PATH such that it points to the root of the MXNet library.

* Build the framework by running:
	```console
	rm -rf build
	mkdir build
	cd build
	cmake ..
	make -j4
	```

* To run the framework, simply type:
	```console
	./COSYNNC
	```

<h1> Configurating the framework for synthesis </h1>
To configure the framework for synthesis...
