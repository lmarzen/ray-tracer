CC = gcc
CFLAGS = -Wall -O3 -lm -std=c99 -fopenmp

.PHONY: clean

raytracer: main.c
	$(CC) main.c -o raytracer $(CFLAGS)

clean:
	rm -f $(OBJECTS) raytracer *.ppm
