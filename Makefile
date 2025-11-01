NVCC_FLAGS  = -O2 -g -Iinclude

SRC_DIR     = src
OBJ_DIR     = obj
BIN_DIR     = bin
TARGET      = $(BIN_DIR)/benchmark

SRC         = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/support.cu
OBJ         = $(OBJ_DIR)/benchmark.o $(OBJ_DIR)/support.o

ifeq ($(OS),Windows_NT)
    NVCC = C:\tools\nvcc.cmd
    MKDIR = @if not exist $(subst /,\, $1) mkdir $(subst /,\, $1)
    RM = @del /Q $(subst /,\, $1) 2>nul || exit 0
    CUDA_PATH ?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v130"
    LD_FLAGS = -lcudart -L"$(CUDA_PATH)\lib\x64"
else
    NVCC = nvcc
    MKDIR = mkdir -p $1
    RM = rm -rf $1
    CUDA_PATH ?= /usr/local/cuda
    LD_FLAGS = -lcudart -L$(CUDA_PATH)/lib64
endif

default: $(TARGET)

$(BIN_DIR) $(OBJ_DIR):
	$(call MKDIR,$@)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJ) | $(BIN_DIR)
	$(NVCC) $(OBJ) -o $@ $(LD_FLAGS)

clean:
	$(call RM,$(OBJ_DIR))
	$(call RM,$(BIN_DIR))