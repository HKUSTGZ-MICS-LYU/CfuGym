CFUGYM_ROOT ?= $(shell git rev-parse --show-toplevel)
TARGET ?= vexii_soc
MARCH ?= rv32imc_zicsr_zifencei

PROJECT_ABS := $(abspath $(CURDIR))
PROJECT_REL := $(patsubst $(abspath $(CFUGYM_ROOT))/%,%,$(PROJECT_ABS))
MAIN_REL := ../$(PROJECT_REL)/main
BUILD_REL := ../$(PROJECT_REL)/build

.PHONY: compile clean

compile:
	$(MAKE) -C $(CFUGYM_ROOT)/sw TARGET=$(TARGET) MAIN=$(MAIN_REL) BUILD=$(BUILD_REL) MARCH=$(MARCH) compile

clean:
	$(MAKE) -C $(CFUGYM_ROOT)/sw TARGET=$(TARGET) MAIN=$(MAIN_REL) BUILD=$(BUILD_REL) MARCH=$(MARCH) clean
