CC = gcc
MPICC = mpicc
# CC = clang
# -std=c99
CFLAGS = -Wall -O3 -lm

.PHONY: sequential openmp mpi clean

sequential:
	$(CC) main.c -o raytracer_sequential $(CFLAGS)
openmp:
	$(CC) main.c -o raytracer_openmp $(CFLAGS) -fopenmp -DUSE_OPENMP
mpi:
	OMPI_CC=$(CC) $(MPICC) main.c -o raytracer_mpi $(CFLAGS) -DUSE_MPI

clean:
	rm -f $(OBJECTS) raytracer_sequential raytracer_openmp raytracer_mpi *.ppm
