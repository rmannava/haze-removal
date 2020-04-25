LIB_DIR = lib
LIB_FILES = $(wildcard $(LIB_DIR)/*.c)

OBJ_DIR = objs

SEQ_BIN = haze_removal_seq
SEQ_DIR = seq
SEQ_FILES = $(wildcard $(SEQ_DIR)/*.c)
SEQ_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SEQ_FILES)) $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_FILES))
SEQ_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SEQ_FILES)) $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_FILES))

CC = gcc
CFLAGS = -O3 -Wall -Werror

CUDA_BIN = haze_removal_cuda
CUDA_DIR = cuda
CUDA_FILES = $(wildcard $(CUDA_DIR)/*.cu)
CUDA_OBJS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(CUDA_FILES)) $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_FILES))

NVCC = nvcc
NVCCFLAGS = -O3 --ptxas-options=-v -arch=sm_30

.PHONY: all dirs clean

all: dirs $(SEQ_BIN) $(CUDA_BIN)

dirs:
	mkdir -p $(OBJ_DIR)/$(LIB_DIR)
	mkdir -p $(OBJ_DIR)/$(SEQ_DIR)
	mkdir -p $(OBJ_DIR)/$(CUDA_DIR)

$(OBJ_DIR)/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

$(SEQ_BIN): $(SEQ_OBJS)
	$(CC) -o $@ $(SEQ_OBJS)

$(CUDA_BIN): $(CUDA_OBJS)
	$(NVCC) -o $@ $(CUDA_OBJS)

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(SEQ_BIN) $(CUDA_BIN)
