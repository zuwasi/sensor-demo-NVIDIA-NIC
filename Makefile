CC=gcc
FLAGS=-O3 -march=native -I.

SRCS=sensor.c

EXEC=sensor.exe

.PHONY = clean all

all : $(EXEC)

$(EXEC) : $(SRCS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	del /F /Q $(EXEC) 2>nul || true

