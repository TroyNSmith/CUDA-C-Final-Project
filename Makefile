CFLAGS = -Wall -fPIC -Iinclude #-std=c99

# Sources: C and optional CUDA
SRC_C = $(wildcard src/*.c)
SRC_CU = $(wildcard src/*.cu)
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
OBJ = $(OBJ_C) $(OBJ_CU)

.PHONY: all clean

ifeq ($(OS),Windows_NT)
# On Windows, prefer nvcc if available (user can override NVCC variable)
NVCC ?= C:\tools\nvcc.cmd
CC = gcc
TARGET = lib/lib.dll
RM_CMD = @cmd /C "del /Q $(subst /,\,$(OBJ)) $(subst /,\,$(TARGET)) 2>nul || exit /B 0"
else
NVCC ?= nvcc
CC = gcc
TARGET = lib/lib.so
RM_CMD = @rm -f $(OBJ) $(TARGET)
endif

# nvcc flags (users can override NVCCFLAGS when calling make)
ifeq ($(OS),Windows_NT)
NVCCFLAGS ?= -Iinclude
else
NVCCFLAGS ?= -Iinclude -Xcompiler -fPIC
endif

all: $(TARGET)

# Create lib dir in a portable way
ifeq ($(OS),Windows_NT)
MKDIR_LIB = @cmd /C "if not exist lib mkdir lib"
else
MKDIR_LIB = mkdir -p lib
endif

# platform libs (avoid -lm on Windows where MSVC linker looks for m.lib)
ifeq ($(OS),Windows_NT)
LIBS =
else
LIBS = -lm
endif

# Link object files into shared library. Use nvcc to link if CUDA objects exist
$(TARGET): $(OBJ)
	$(MKDIR_LIB)

ifeq ($(strip $(OBJ_CU)),)
	$(CC) -shared -o $@ $^ $(LIBS)
else
	$(NVCC) -shared $(NVCCFLAGS) -o $@ $^ $(LIBS)
endif

# Compile .c into .o
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile .cu into .o using nvcc
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM_CMD)