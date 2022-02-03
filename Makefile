# Copyright (c) 2022 RidgeRun,LLC <support@ridgerun.com>

# If OpenVX is installed in a non-standard location, set the
# appropriate flags here or via cmdline as
#
#   make VX_CFLAGS=-I/path/to/include VX_LDFLAGS=-L/path/to/libs
#
VX_CFLAGS?=
VX_LDFLAGS?=

SOURCES=$(wildcard vx_training_*.c)
SOURCES_CC=$(wildcard vx_training_*.cc)

PROGRAMS=$(patsubst %.c,%,$(SOURCES))
PROGRAMS_CC=$(patsubst %.cc,%,$(SOURCES_CC))

.PHONY: all clean

all: $(PROGRAMS) $(PROGRAMS_CC)

%: %.cc Makefile
	@printf "Building $@ from $< - "
	@$(CXX) -o $@ $< -g -O0 $(VX_CFLAGS) $(CFLAGS) $(VX_LDFLAGS) $(LD_FLAGS) -lopenvx -lm `pkg-config --cflags --libs opencv4` -std=c++11
	@echo " done!"

%: %.c Makefile
	@printf "Building $@ from $< - "
	@$(CC) -o $@ $< -g -O0 $(VX_CFLAGS) $(CFLAGS) $(VX_LDFLAGS) $(LD_FLAGS) -lopenvx -lm
	@echo " done!"

clean:
	@rm -f *~ $(PROGRAMS)
