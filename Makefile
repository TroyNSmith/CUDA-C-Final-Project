CC = gcc
CFLAGS = -std=c99 -Wall -fPIC -Iinclude
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
TARGET = lib/lib.so

all: $(TARGET)

$(TARGET): $(OBJ)
	mkdir -p lib
	$(CC) -shared -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) lib/*.so