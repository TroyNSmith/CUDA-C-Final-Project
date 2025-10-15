CC = gcc
CFLAGS = -std=c99 -Wall -fPIC -Iinclude
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
TARGET = lib/lib.so

all: $(TARGET)

.PHONY: all clean

ifeq ($(OS),Windows_NT)
RM_CMD = @cmd /C "del /Q $(subst /,\\,$(OBJ)) $(subst /,\\,$(TARGET)) 2>nul || exit /B 0"
else
RM_CMD = @rm -f $(OBJ) $(TARGET)
endif

# Rule to link object files into shared library
$(TARGET): $(OBJ)
	mkdir -p lib
	$(CC) -shared -o $@ $^ -lm

# Rule to compile .c into .o
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM_CMD)