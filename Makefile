CC = gcc
CFLAGS = -std=c99 -Wall -fPIC -Iinclude
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
TARGET = lib/lib.so

all: $(TARGET)

# Rule to link object files into shared library
$(TARGET): $(OBJ)
	mkdir -p lib
	$(CC) -shared -o $@ $^ -lm

# Rule to compile .c into .o
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)