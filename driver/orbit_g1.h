/* SPDX-License-Identifier: GPL-2.0 */
/*
 * ORBIT-G1 PCIe Accelerator Driver — Internal Header
 *
 * This file is NOT exported to userspace. It defines the kernel-internal
 * data structures, register offsets, and macros used across driver source files.
 */
#ifndef _ORBIT_G1_H
#define _ORBIT_G1_H

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/atomic.h>
#include <linux/list.h>
#include <linux/debugfs.h>
#include <linux/interrupt.h>

#include "../include/uapi/orbit_g1_uapi.h"

/* =========================================================================
 * PCI identification
 * ========================================================================= */
#define ORBIT_G1_VENDOR_ID		0x1234
#define ORBIT_G1_DEVICE_ID		0x0001

/* =========================================================================
 * BAR indices
 * ========================================================================= */
#define ORBIT_BAR0			0	/* descriptor queue registers */
#define ORBIT_BAR1			1	/* GDDR6 device memory window  */

/* =========================================================================
 * BAR0 Global Register Offsets
 * ========================================================================= */
#define ORBIT_REG_ID			0x0000	/* RO: magic 0x4F524231 "ORB1" */
#define ORBIT_REG_HW_REV		0x0004	/* RO: hardware revision */
#define ORBIT_REG_FW_VER		0x0008	/* RO: firmware version */
#define ORBIT_REG_STATUS		0x000C	/* RO: global status */
#define ORBIT_REG_GDDR_SIZE_LO		0x0010	/* RO: GDDR6 size [31:0] */
#define ORBIT_REG_GDDR_SIZE_HI		0x0014	/* RO: GDDR6 size [63:32] */
#define ORBIT_REG_GLOBAL_CTRL		0x0018	/* RW: soft reset (bit 0) */
#define ORBIT_REG_INTR_STATUS		0x001C	/* RW1C: per-queue IRQ status */
#define ORBIT_REG_INTR_MASK		0x0020	/* RW: per-queue IRQ mask */

/* ORBIT_REG_ID expected magic value */
#define ORBIT_REG_ID_MAGIC		0x4F524231U	/* "ORB1" */

/* ORBIT_REG_STATUS bits */
#define ORBIT_STATUS_READY		BIT(0)
#define ORBIT_STATUS_FAULT		BIT(1)
#define ORBIT_STATUS_BUSY		BIT(2)

/* ORBIT_REG_GLOBAL_CTRL bits */
#define ORBIT_CTRL_SOFT_RESET		BIT(0)
#define ORBIT_CTRL_SUP_MODE_SHIFT	2
#define ORBIT_CTRL_SUP_MODE_MASK	(0x3U << ORBIT_CTRL_SUP_MODE_SHIFT)

/* =========================================================================
 * BAR0 Per-Queue Register Block
 *
 * Base = 0x0100 + (queue_id * 0x40)
 * ========================================================================= */
#define ORBIT_Q_BLOCK_BASE		0x0100
#define ORBIT_Q_BLOCK_STRIDE		0x40

#define ORBIT_Q_RING_BASE_LO_OFF	0x00	/* RW: ring DMA addr [31:0] */
#define ORBIT_Q_RING_BASE_HI_OFF	0x04	/* RW: ring DMA addr [63:32] */
#define ORBIT_Q_RING_SIZE_OFF		0x08	/* RW: ring capacity (entries) */
#define ORBIT_Q_HEAD_OFF		0x0C	/* RO: hardware consumer head */
#define ORBIT_Q_TAIL_OFF		0x10	/* RW: software producer tail (doorbell) */
#define ORBIT_Q_STATUS_OFF		0x14	/* RO: queue status */
#define ORBIT_Q_ERROR_CODE_OFF		0x18	/* RO: last error code */
#define ORBIT_Q_COMPLETE_CNT_OFF	0x1C	/* RO: completed count (wrapping) */
#define ORBIT_Q_INTR_VECTOR_OFF		0x20	/* RW: MSI-X vector assignment */
#define ORBIT_Q_CONFIG_OFF		0x24	/* RW: queue config flags */

/* Compute per-queue block base for a given queue_id (0-3) */
#define ORBIT_Q_BLOCK(q)		(ORBIT_Q_BLOCK_BASE + (q) * ORBIT_Q_BLOCK_STRIDE)

/* Full per-queue register addresses */
#define ORBIT_Q_RING_BASE_LO(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_RING_BASE_LO_OFF)
#define ORBIT_Q_RING_BASE_HI(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_RING_BASE_HI_OFF)
#define ORBIT_Q_RING_SIZE(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_RING_SIZE_OFF)
#define ORBIT_Q_HEAD(q)			(ORBIT_Q_BLOCK(q) + ORBIT_Q_HEAD_OFF)
#define ORBIT_Q_TAIL(q)			(ORBIT_Q_BLOCK(q) + ORBIT_Q_TAIL_OFF)
#define ORBIT_Q_STATUS(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_STATUS_OFF)
#define ORBIT_Q_ERROR_CODE(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_ERROR_CODE_OFF)
#define ORBIT_Q_COMPLETE_CNT(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_COMPLETE_CNT_OFF)
#define ORBIT_Q_INTR_VECTOR(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_INTR_VECTOR_OFF)
#define ORBIT_Q_CONFIG(q)		(ORBIT_Q_BLOCK(q) + ORBIT_Q_CONFIG_OFF)

/* Queue status bits */
#define ORBIT_Q_STATUS_IDLE		BIT(0)
#define ORBIT_Q_STATUS_ACTIVE		BIT(1)
#define ORBIT_Q_STATUS_ERROR		BIT(2)

/* =========================================================================
 * Register Accessor macros (operate on struct orbit_g1_device *)
 * The inline 64-bit helpers are defined below, after struct orbit_g1_device.
 * ========================================================================= */
#define orbit_read32(dev, off)		ioread32((dev)->bar0 + (off))
#define orbit_write32(dev, off, val)	iowrite32((val), (dev)->bar0 + (off))

/* =========================================================================
 * Memory pool / buddy allocator
 * ========================================================================= */
#define ORBIT_MEM_MIN_ORDER		12	/* 4 KB minimum allocation */
#define ORBIT_MEM_MAX_ORDER		34	/* 16 GB = 2^34 bytes */
#define ORBIT_MEM_NUM_ORDERS		(ORBIT_MEM_MAX_ORDER - ORBIT_MEM_MIN_ORDER + 1)

struct orbit_mem_block {
	struct list_head	list;
	u64			base;		/* GDDR6-physical address */
	u32			order;		/* 2^order bytes */
};

struct orbit_mem_pool {
	u64			base;		/* GDDR6 base (usually 0) */
	u64			size;		/* total GDDR6 size in bytes */
	spinlock_t		lock;
	struct list_head	free_lists[ORBIT_MEM_NUM_ORDERS];
	struct list_head	alloc_list;	/* list of allocated orbit_mem_block structs */
	u64			allocated_bytes;
	u32			min_order;
	u32			max_order;
};

/* =========================================================================
 * Descriptor Ring Queue
 * ========================================================================= */
struct orbit_queue {
	void			*ring_cpu;	/* kernel VA (DMA coherent) */
	dma_addr_t		ring_dma;	/* bus address → Q_RING_BASE */
	u32			ring_size;	/* entries (256) */
	u32			tail;		/* next slot to write (SW-owned) */
	u32			head_cached;	/* cached snapshot of HW head */
	spinlock_t		lock;		/* protect tail update */
	wait_queue_head_t	wq;		/* wait_event for completion */
	atomic_t		inflight;	/* submitted but not completed */
	u64			next_cookie;	/* next submit_cookie value */
	atomic64_t		done_cookie;	/* highest completed cookie (atomic64 for 32-bit safety) */
	u32			queue_id;
	struct orbit_g1_device	*dev;
};

/* =========================================================================
 * Per-fd (open file) context
 * ========================================================================= */
struct orbit_fd_ctx {
	struct orbit_g1_device	*dev;
	spinlock_t		alloc_lock;
	struct list_head	alloc_list;	/* GDDR6 regions owned by this fd */
	u32			session_id;	/* unique per open() */
};

/* Entry in the per-fd allocation list */
struct orbit_alloc_entry {
	struct list_head	list;
	u64			handle;		/* == base GDDR6 addr, used as opaque handle */
	u64			device_addr;
	u64			size_bytes;
};

/* =========================================================================
 * Main device structure
 * ========================================================================= */
struct orbit_g1_device {
	struct pci_dev		*pdev;
	void __iomem		*bar0;		/* descriptor queue registers */
	void __iomem		*bar1;		/* GDDR6 window */
	resource_size_t		bar0_len;
	resource_size_t		bar1_len;

	struct orbit_queue	queues[ORBIT_NUM_QUEUES];
	struct orbit_mem_pool	gddr_pool;

	int			msix_nvec;
	/* pci_alloc_irq_vectors API — no msix_entry array needed for modern kernels */

	struct cdev		cdev;
	struct class		*class;
	struct device		*dev;
	dev_t			devt;
	int			minor;

	spinlock_t		lock;		/* protect device-wide state */
	atomic_t		open_count;

	u64			gddr_size_bytes;
	u32			fw_version;
	u32			hw_revision;

	struct dentry		*debugfs_dir;
};

/* =========================================================================
 * Register Accessor inline helpers (64-bit, require struct orbit_g1_device)
 * ========================================================================= */

/* 64-bit read via two 32-bit accesses (lo word first) */
static inline u64 orbit_read64(struct orbit_g1_device *dev, u32 off)
{
	u64 lo = orbit_read32(dev, off);
	u64 hi = orbit_read32(dev, off + 4);

	return lo | (hi << 32);
}

static inline void orbit_write64(struct orbit_g1_device *dev, u32 off, u64 val)
{
	orbit_write32(dev, off,     (u32)(val & 0xFFFFFFFFULL));
	orbit_write32(dev, off + 4, (u32)(val >> 32));
}

/* =========================================================================
 * Module-wide globals (defined in orbit_g1_pci.c)
 * ========================================================================= */
extern struct class *orbit_g1_class;
extern dev_t orbit_g1_devt_base;

/* =========================================================================
 * Function prototypes — cross-file interfaces
 * ========================================================================= */

/* orbit_g1_pci.c */
int  orbit_g1_mmio_init(struct orbit_g1_device *odev);
void orbit_g1_mmio_fini(struct orbit_g1_device *odev);

/* orbit_g1_queue.c */
int  orbit_g1_queue_init(struct orbit_g1_device *odev);
void orbit_g1_queue_fini(struct orbit_g1_device *odev);
int  orbit_queue_submit(struct orbit_queue *q,
			const void *descs, u32 count, u64 *cookie_out);
int  orbit_queue_wait(struct orbit_queue *q, u64 cookie, u32 timeout_ms);
void orbit_queue_complete(struct orbit_g1_device *odev, int qid);
int  orbit_queue_reset(struct orbit_g1_device *odev, u32 qid);

/* orbit_g1_mem.c */
int  orbit_g1_mem_init(struct orbit_g1_device *odev);
void orbit_g1_mem_fini(struct orbit_g1_device *odev);
int  orbit_mem_alloc(struct orbit_mem_pool *pool,
		     u64 size_bytes, u64 align_bytes,
		     u64 *device_addr_out, u64 *handle_out);
int  orbit_mem_free(struct orbit_mem_pool *pool, u64 handle);

/* orbit_g1_chardev.c */
int  orbit_g1_cdev_create(struct orbit_g1_device *odev);
void orbit_g1_cdev_destroy(struct orbit_g1_device *odev);

/* IRQ handlers (defined in orbit_g1_pci.c, triggered from MSI-X) */
irqreturn_t orbit_irq_q0(int irq, void *data);
irqreturn_t orbit_irq_q1(int irq, void *data);
irqreturn_t orbit_irq_q2(int irq, void *data);
irqreturn_t orbit_irq_q3(int irq, void *data);
irqreturn_t orbit_irq_common(struct orbit_g1_device *dev, int qid);

#endif /* _ORBIT_G1_H */
