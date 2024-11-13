CC = gcc
CFLAGS = -Wall -Wextra -I./src
SRC_DIR = src
MODELS_DIR = $(SRC_DIR)/models
OBJ_DIR = obj

TARGET = auto_dl_install
DIST_DIR = dist

SRCS = $(SRC_DIR)/main.c \
       $(MODELS_DIR)/applio/scripts.c \
       $(MODELS_DIR)/fish_speech/scripts.c

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(TARGET) package

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@

package:
	mkdir -p $(DIST_DIR)
	cp $(TARGET) $(DIST_DIR)/
	chmod +x $(DIST_DIR)/$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)
	rm -rf $(DIST_DIR)

.PHONY: all install clean package
