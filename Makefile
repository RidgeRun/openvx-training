# Copyright (c) 2022 RidgeRun,LLC <support@ridgerun.com>

# If OpenVX is installed in a non-standard location, set the
# appropriate flags here or via cmdline as
#
#   make VX_CFLAGS=-I/path/to/include VX_LDFLAGS=-L/path/to/libs
#
VX_CFLAGS=
VX_LDFLAGS=

SOURCES=$(wildcard vx_training_*.c)
PROGRAMS=$(patsubst %.c,%,$(SOURCES))

.PHONY: all clean

all: $(PROGRAMS)

%: %.c Makefile
	@printf "Buildig $@ from $< - "
	@$(CC) -o $@ $< $(VX_CFLAGS) $(CFLAGS) $(VX_LDFLAGS) $(LD_FLAGS) -lopenvx
	@echo " done!"

clean:
	@rm -f *~ $(PROGRAMS)
