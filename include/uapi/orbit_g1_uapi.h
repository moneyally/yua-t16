/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * ORBIT-G1 Userspace API Header
 *
 * This file is exported to userspace. It defines all ioctl structures and
 * numbers for the ORBIT-G1 PCIe accelerator driver.
 *
 * All ioctl magic: 'O' (0x4F)
 */
#ifndef _UAPI_ORBIT_G1_H
#define _UAPI_ORBIT_G1_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* -------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */

#define ORBIT_NUM_QUEUES		4
#define ORBIT_QUEUE_DEPTH		256
#define ORBIT_DESC_SIZE			64		/* bytes per descriptor */

/* Descriptor type codes (v1 + v2) */
#define ORBIT_DESC_DMA_2D		0x01
#define ORBIT_DESC_GEMM_INT8		0x02
#define ORBIT_DESC_VECTOR_OP		0x03
#define ORBIT_DESC_COPY_2D_PLUS		0x04
#define ORBIT_DESC_FORMAT_CONVERT	0x05
#define ORBIT_DESC_FINGERPRINT		0x06
#define ORBIT_DESC_BARRIER		0x07
#define ORBIT_DESC_EVENT		0x08
#define ORBIT_DESC_PERF_SNAPSHOT	0x09
/* v2 extensions */
#define ORBIT_DESC_KVC_READ		0x0A
#define ORBIT_DESC_KVC_WRITE		0x0B
#define ORBIT_DESC_MOE_ROUTE		0x0C
#define ORBIT_DESC_VECTOR_OP_EX		0x0D
#define ORBIT_DESC_GEMM_INT4		0x0E
#define ORBIT_DESC_SOFTMAX		0x0F

/* ORBIT_IOC_SUBMIT_DESC flags */
#define ORBIT_SUBMIT_WAIT		(1U << 0)	/* block until completion */
#define ORBIT_SUBMIT_NOWAIT		(1U << 1)	/* return -ENOSPC if ring full */

/* Queue IDs */
#define ORBIT_QUEUE_COMPUTE		0
#define ORBIT_QUEUE_UTILITY		1
#define ORBIT_QUEUE_TELEMETRY		2
#define ORBIT_QUEUE_HIPRI		3

/* -------------------------------------------------------------------------
 * ORBIT_IOC_SUBMIT_DESC — submit a batch of descriptors to a queue
 *
 * _IOWR('O', 0x01, struct orbit_desc_submit)
 *
 * On success: submit_cookie is filled with a monotonic u64 token that can
 * be used with ORBIT_IOC_WAIT_DONE.
 * ------------------------------------------------------------------------- */
struct orbit_desc_submit {
	__u32		queue_id;	/* target queue (0-3) */
	__u32		count;		/* number of 64-byte descriptors */
	__u32		flags;		/* ORBIT_SUBMIT_* */
	__u32		_pad;
	__u64		descs_ptr;	/* userspace pointer to desc array */
	__u64		submit_cookie;	/* OUT: opaque completion token */
};

/* -------------------------------------------------------------------------
 * ORBIT_IOC_WAIT_DONE — wait for a submitted batch to complete
 *
 * _IOWR('O', 0x02, struct orbit_wait_done)
 *
 * Returns 0 on completion, -ETIMEDOUT on timeout, -ERESTARTSYS on signal.
 * ------------------------------------------------------------------------- */
struct orbit_wait_done {
	__u32		queue_id;	/* queue the cookie was issued on */
	__u32		timeout_ms;	/* 0 = wait forever */
	__u64		submit_cookie;	/* token from ORBIT_IOC_SUBMIT_DESC */
};

/* -------------------------------------------------------------------------
 * ORBIT_IOC_ALLOC_MEM — allocate a region from GDDR6 memory pool
 *
 * _IOWR('O', 0x03, struct orbit_mem_alloc)
 *
 * On success: device_addr and handle are filled. handle is an opaque 64-bit
 * token that must be passed to ORBIT_IOC_FREE_MEM to release the region.
 * ------------------------------------------------------------------------- */
struct orbit_mem_alloc {
	__u64		size_bytes;	/* IN: requested size */
	__u64		align_bytes;	/* IN: alignment (must be power-of-2, >= 4096) */
	__u64		device_addr;	/* OUT: GDDR6-physical base address */
	__u64		handle;		/* OUT: opaque allocation token */
};

/* -------------------------------------------------------------------------
 * ORBIT_IOC_FREE_MEM — free a GDDR6 region
 *
 * _IOW('O', 0x04, struct orbit_mem_free)
 * ------------------------------------------------------------------------- */
struct orbit_mem_free {
	__u64		handle;		/* token from ORBIT_IOC_ALLOC_MEM */
};

/* -------------------------------------------------------------------------
 * ORBIT_IOC_GET_INFO — query device capabilities
 *
 * _IOR('O', 0x05, struct orbit_device_info)
 * ------------------------------------------------------------------------- */
struct orbit_device_info {
	__u64		gddr_size_bytes;	/* total GDDR6 size */
	__u64		bar1_size_bytes;	/* BAR1 mmap window size */
	__u32		num_queues;		/* always ORBIT_NUM_QUEUES */
	__u32		queue_depth;		/* entries per queue */
	__u32		fw_version;		/* firmware version */
	__u32		hw_revision;		/* hardware revision */
	__u32		desc_spec_version;	/* descriptor spec version */
	__u32		supported_desc_types;	/* bitmask of ORBIT_DESC_* */
	__u8		device_name[32];	/* null-terminated product string */
	__u8		_pad[4];
};

/* -------------------------------------------------------------------------
 * ORBIT_IOC_RESET_QUEUE — drain and reset a queue (error recovery)
 *
 * _IOW('O', 0x06, struct orbit_queue_reset)
 * ------------------------------------------------------------------------- */
struct orbit_queue_reset {
	__u32		queue_id;	/* queue to reset (0-3) */
	__u32		flags;		/* reserved, must be 0 */
};

/* -------------------------------------------------------------------------
 * ioctl numbers
 * ------------------------------------------------------------------------- */
#define ORBIT_IOC_MAGIC		'O'

#define ORBIT_IOC_SUBMIT_DESC	_IOWR(ORBIT_IOC_MAGIC, 0x01, struct orbit_desc_submit)
#define ORBIT_IOC_WAIT_DONE	_IOWR(ORBIT_IOC_MAGIC, 0x02, struct orbit_wait_done)
#define ORBIT_IOC_ALLOC_MEM	_IOWR(ORBIT_IOC_MAGIC, 0x03, struct orbit_mem_alloc)
#define ORBIT_IOC_FREE_MEM	_IOW(ORBIT_IOC_MAGIC,  0x04, struct orbit_mem_free)
#define ORBIT_IOC_GET_INFO	_IOR(ORBIT_IOC_MAGIC,  0x05, struct orbit_device_info)
#define ORBIT_IOC_RESET_QUEUE	_IOW(ORBIT_IOC_MAGIC,  0x06, struct orbit_queue_reset)

#endif /* _UAPI_ORBIT_G1_H */
