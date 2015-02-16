CC = gcc
CFLAGS = -Wall -Wextra -Werror -pedantic -std=c99 -O3
LDFLAGS = -O3 -lm
TARGETS = neural cuda

all: $(TARGETS)

neural: main.o mnist.o network.o neuron.o
	$(CC) $(LDFLAGS) $^ -o $@

cuda: cuda.o mnist.o
	$(CC) -O3 -lcudart $^ -o $@

%.o: %.cu
	nvcc -c -arch=sm_30 $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o

destroy: clean
	rm -f $(TARGETS)

rebuild: destroy all
