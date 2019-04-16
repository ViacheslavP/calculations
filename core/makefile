
all:
	/home/pviacheslav/.intel/intelpython3/bin/cython -3 -o sigmaMatrix.c sigmaMatrix.pyx
	gcc-6 -g -O3 -fpic -c sigmaMatrix.c -o sigmaMatrix.o `/home/pviacheslav/.intel/intelpython3/bin/python3-config --cflags`
	gcc-6 -g -O3 -shared -o sigmaMatrix.so sigmaMatrix.o `/home/pviacheslav/.intel/intelpython3/bin/python3-config --libs`
