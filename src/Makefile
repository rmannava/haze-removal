LIB_DIR = lib
LIB_FILES = $(wildcard $(LIB_DIR)/*.c)

OBJ_DIR = objs

SEQ_BIN = haze_removal_seq
SEQ_DIR = seq
SEQ_FILES = $(wildcard $(SEQ_DIR)/*.c)
SEQ_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SEQ_FILES)) $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_FILES))

CC = gcc
CFLAGS = -O3 -Wall -Werror

.PHONY: all

all: dirs $(SEQ_BIN)

dirs:
	mkdir -p $(OBJ_DIR)/$(LIB_DIR)
	mkdir -p $(OBJ_DIR)/$(SEQ_DIR)

$(OBJ_DIR)/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(SEQ_BIN): $(SEQ_OBJS)
	$(CC) -o $(SEQ_BIN) $(SEQ_OBJS)

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(SEQ_BIN)
