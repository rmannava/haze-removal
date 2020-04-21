SEQ_BIN = haze_removal_seq
SEQ_DIR = seq
SEQ_FILES = $(SEQ_DIR)/main.c
SEQ_OBJDIR = seq_objs
SEQ_OBJS = $(SEQ_OBJDIR)/main.o $(SEQ_OBJDIR)/$(LIBDIR)/lodepng.o

CC = gcc
CFLAGS = -g -Wall -Werror

LIBDIR = lib
LIBFILES = $(LIBDIR)/lodepng.c

.PHONY: all

all: dirs $(SEQ_BIN)

dirs:
	mkdir -p $(SEQ_OBJDIR)
	mkdir -p $(SEQ_OBJDIR)/$(LIBDIR)

$(SEQ_OBJDIR)/%.o: $(SEQ_DIR)/%.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(SEQ_OBJDIR)/$(LIBDIR)/%.o: $(LIBDIR)/%.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(SEQ_BIN): $(SEQ_OBJS)
	$(CC) -o $(SEQ_BIN) $(SEQ_OBJS)

clean:
	rm -rf $(SEQ_OBJDIR)
	rm -f $(SEQ_BIN)
