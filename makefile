all:
	/home/viacheslav/.anaconda/bin/cython -3 -o sigmaMatrix.c sigmaMatrix.pyx
	gcc -g -O2 -fpic -c sigmaMatrix.c -o sigmaMatrix.o `/home/viacheslav/.anaconda/bin/python3-config --cflags`
	gcc -g -O2 -shared -o sigmaMatrix.so sigmaMatrix.o `~/home/viacheslav/.anaconda/bin/python3-config --libs`
