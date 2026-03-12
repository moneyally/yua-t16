# SPDX-License-Identifier: GPL-2.0
#
# ORBIT-G1 PCIe Accelerator Kernel Module — Makefile
#
# Usage:
#   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
#   make -C /lib/modules/$(uname -r)/build M=$(pwd) clean
#
# Optional overrides:
#   KDIR=/path/to/kernel/source  — use a custom kernel source tree
#   CCFLAGS_MODULE               — extra compiler flags (e.g. -DDEBUG)

# --------------------------------------------------------------------------
# Kernel build directory
# --------------------------------------------------------------------------
KDIR ?= /lib/modules/$(shell uname -r)/build

# --------------------------------------------------------------------------
# Module object composition
# --------------------------------------------------------------------------
obj-m := orbit_g1.o

orbit_g1-objs := \
	orbit_g1_pci.o     \
	orbit_g1_queue.o   \
	orbit_g1_mem.o     \
	orbit_g1_chardev.o

# --------------------------------------------------------------------------
# Include paths
# --------------------------------------------------------------------------
# The uapi header lives one level up under include/uapi/.
# ccflags-y is the modern way to pass -I flags (not EXTRA_CFLAGS).
ccflags-y += -I$(src)/..

# Optional: enable verbose debugging output
# ccflags-y += -DDEBUG

# --------------------------------------------------------------------------
# Build targets
# --------------------------------------------------------------------------
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

install:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules_install
	depmod -A

# --------------------------------------------------------------------------
# Quick syntax-check target (does not link, just compiles)
# --------------------------------------------------------------------------
check:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules \
		EXTRA_CFLAGS="-W -Wextra -Werror"

.PHONY: all clean install check
