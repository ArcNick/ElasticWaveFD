CC = nvcc
CFLAGS = -rdc=true -I include -L./lib -lcjson -std=c++17 -lm
SRC = src/*.cu
OUT = bin/main_debug

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)