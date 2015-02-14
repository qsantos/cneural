CC = gcc
CFLAGS = -Wall -Wextra -Werror -pedantic -std=c99 -O3
LDFLAGS = -O3 -lm
TARGETS = neural

all: $(TARGETS)

neural: main.o network.o neuron.o
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o

destroy: clean
	rm -f $(TARGETS)

rebuild: destroy all
