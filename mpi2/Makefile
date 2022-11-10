CC = clang
MPICC = mpicc
CFLAGS = -Wall -O3 -lm -std=c99

.PHONY: clean

raytracer: main.c
	OMPI_CC=$(CC) $(MPICC) main.c -o raytracer $(CFLAGS)

clean:
	rm -f $(OBJECTS) raytracer *.ppm
