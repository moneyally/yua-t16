// SPDX-License-Identifier: GPL-2.0
/*
 * ORBIT-G1 PCIe Accelerator Driver
 * orbit_g1_queue.c — Descriptor ring buffer: enqueue, doorbell, completion ISR
 */
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/jiffies.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/uaccess.h>

#include "orbit_g1.h"

/* =========================================================================
 * Queue init / fini
 * ========================================================================= */

/**
 * orbit_g1_queue_init - Initialize software state for all 4 queues.
 *
 * DMA memory for the rings must already be allocated (orbit_g1_dma_init).
 * This function programs the ring base address and size into hardware,
 * and initialises all software bookkeeping.
 */
int orbit_g1_queue_init(struct orbit_g1_device *odev)
{
	int i;

	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		struct orbit_queue *q = &odev->queues[i];

		/* Software state */
		q->queue_id	= i;
		q->dev		= odev;
		q->tail		= 0;
		q->head_cached	= 0;
		q->next_cookie	= 1;
		atomic64_set(&q->done_cookie, 0);
		spin_lock_init(&q->lock);
		init_waitqueue_head(&q->wq);
		atomic_set(&q->inflight, 0);

		/* Program ring base address and size into BAR0 */
		orbit_write32(odev, ORBIT_Q_RING_BASE_LO(i),
			      (u32)(q->ring_dma & 0xFFFFFFFFULL));
		orbit_write32(odev, ORBIT_Q_RING_BASE_HI(i),
			      (u32)(q->ring_dma >> 32));
		orbit_write32(odev, ORBIT_Q_RING_SIZE(i), q->ring_size);

		/* Reset head/tail to zero */
		orbit_write32(odev, ORBIT_Q_TAIL(i), 0);

		dev_dbg(&odev->pdev->dev,
			"Queue %d init: ring_dma=0x%llx size=%u\n",
			i, (u64)q->ring_dma, q->ring_size);
	}

	return 0;
}

/**
 * orbit_g1_queue_fini - Tear down queue software state.
 */
void orbit_g1_queue_fini(struct orbit_g1_device *odev)
{
	int i;

	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		struct orbit_queue *q = &odev->queues[i];

		/* Wake up anyone waiting so they can observe the teardown */
		wake_up_all(&q->wq);
	}
}

/* =========================================================================
 * Submission path
 * ========================================================================= */

/**
 * orbit_queue_submit - Copy descriptors into the ring and ring the doorbell.
 *
 * @q:          target queue
 * @descs:      kernel buffer containing @count × ORBIT_DESC_SIZE bytes
 * @count:      number of descriptors to submit
 * @cookie_out: on success, filled with a monotonic completion token
 *
 * Returns 0 on success, -ENOSPC if the ring is full, or other negative errno.
 *
 * The caller must hold no locks. This function acquires q->lock internally.
 */
int orbit_queue_submit(struct orbit_queue *q,
		       const void *descs, u32 count, u64 *cookie_out)
{
	struct orbit_g1_device *odev = q->dev;
	const u8 *src = (const u8 *)descs;
	u32 i, avail, head_fresh;
	unsigned long flags;
	u64 cookie;

	if (count == 0 || count > q->ring_size)
		return -EINVAL;

	spin_lock_irqsave(&q->lock, flags);

	/* Check available slots using cached head */
	avail = q->ring_size - (q->tail - q->head_cached);
	if (avail < count) {
		/* Refresh head from hardware */
		head_fresh = orbit_read32(odev, ORBIT_Q_HEAD(q->queue_id));
		q->head_cached = head_fresh;
		avail = q->ring_size - (q->tail - q->head_cached);
	}

	if (avail < count) {
		spin_unlock_irqrestore(&q->lock, flags);
		return -ENOSPC;
	}

	/* Copy descriptors into the ring (wrapping) */
	for (i = 0; i < count; i++) {
		u32 slot = (q->tail + i) % q->ring_size;
		u8 *dst = (u8 *)q->ring_cpu + (slot * ORBIT_DESC_SIZE);

		memcpy(dst, src + (i * ORBIT_DESC_SIZE), ORBIT_DESC_SIZE);
	}

	/* Assign cookie before incrementing tail */
	cookie = q->next_cookie;
	q->next_cookie += count;

	/* Ensure all descriptor writes are visible to the device */
	wmb();

	/* Advance the tail — this is the doorbell */
	q->tail += count;
	orbit_write32(odev, ORBIT_Q_TAIL(q->queue_id), q->tail);

	atomic_add(count, &q->inflight);

	spin_unlock_irqrestore(&q->lock, flags);

	if (cookie_out)
		*cookie_out = cookie + count - 1;	/* last descriptor's cookie */

	return 0;
}

/* =========================================================================
 * Completion path (called from ISR)
 * ========================================================================= */

/**
 * orbit_queue_complete - Process completions for queue @qid.
 *
 * Called from the MSI-X ISR. Reads the hardware head pointer, calculates
 * how many descriptors completed, and wakes up waiters.
 *
 * Runs in interrupt context — must not sleep.
 */
void orbit_queue_complete(struct orbit_g1_device *odev, int qid)
{
	struct orbit_queue *q = &odev->queues[qid];
	u32 new_head;
	u32 completed;

	new_head = orbit_read32(odev, ORBIT_Q_HEAD(qid));

	/*
	 * Wrap-around protection: the hardware head counter is 32-bit and
	 * wraps at 2^32.  Compute completed using unsigned subtraction so the
	 * result is correct across the wrap boundary.
	 *
	 * Example: head_cached = 0xFFFFFFFF, new_head = 0x00000002
	 *   completed = (u32)(0x00000002 - 0xFFFFFFFF) = 3  ✓
	 */
	completed = new_head - q->head_cached;	/* unsigned 32-bit wrap-safe */

	if (completed == 0)
		return;

	q->head_cached = new_head;

	/*
	 * done_cookie tracks the highest completed descriptor serial.
	 * cookie == head position (simplified).
	 * atomic64_t ensures a correct 64-bit write on 32-bit architectures
	 * that lack 64-bit store atomicity.
	 */
	atomic64_set(&q->done_cookie, (u64)new_head);

	atomic_sub((int)completed, &q->inflight);

	/* Wake all waiters — they will check their specific cookie */
	wake_up_all(&q->wq);
}

/* =========================================================================
 * Wait path
 * ========================================================================= */

/**
 * orbit_queue_wait - Sleep until @cookie's descriptor completes or timeout.
 *
 * @q:          queue the cookie belongs to
 * @cookie:     submit_cookie from orbit_queue_submit()
 * @timeout_ms: milliseconds to wait; 0 means wait forever
 *
 * Returns:
 *   0            — completed successfully
 *  -ETIMEDOUT   — timed out
 *  -ERESTARTSYS — interrupted by signal
 */
int orbit_queue_wait(struct orbit_queue *q, u64 cookie, u32 timeout_ms)
{
	long ret;
	unsigned long timeout_jiffies;

	if (timeout_ms == 0) {
		ret = wait_event_interruptible(q->wq,
					       atomic64_read(&q->done_cookie) >= (s64)cookie);
		if (ret == -ERESTARTSYS)
			return -ERESTARTSYS;
		return 0;
	}

	timeout_jiffies = msecs_to_jiffies(timeout_ms);
	ret = wait_event_interruptible_timeout(q->wq,
					       atomic64_read(&q->done_cookie) >= (s64)cookie,
					       timeout_jiffies);
	if (ret < 0)
		return -ERESTARTSYS;
	if (ret == 0)
		return -ETIMEDOUT;
	return 0;
}

/* =========================================================================
 * Queue reset (error recovery)
 * ========================================================================= */

/**
 * orbit_queue_reset - Drain inflight work and reset a queue to idle state.
 *
 * This is a best-effort drain: it waits up to 1 second for inflight
 * descriptors to complete, then forces a software reset of the queue's
 * head/tail counters.
 *
 * Called from ORBIT_IOC_RESET_QUEUE ioctl with process context.
 */
int orbit_queue_reset(struct orbit_g1_device *odev, u32 qid)
{
	struct orbit_queue *q;
	unsigned long flags;
	int ret;

	if (qid >= ORBIT_NUM_QUEUES)
		return -EINVAL;

	q = &odev->queues[qid];

	/*
	 * TODO: Issue a hardware queue flush command via Q_CONFIG if
	 * the hardware supports it.  For now, wait for inflight to drain.
	 */
	ret = wait_event_interruptible_timeout(q->wq,
					       atomic_read(&q->inflight) == 0,
					       msecs_to_jiffies(1000));
	if (ret == 0)
		dev_warn(&odev->pdev->dev,
			 "Queue %u reset: timed out waiting for drain\n", qid);

	spin_lock_irqsave(&q->lock, flags);

	/* Reset software pointers */
	q->tail		= 0;
	q->head_cached	= 0;
	q->next_cookie	= 1;
	atomic64_set(&q->done_cookie, 0);
	atomic_set(&q->inflight, 0);

	/* Reset hardware pointers */
	orbit_write32(odev, ORBIT_Q_TAIL(qid), 0);

	spin_unlock_irqrestore(&q->lock, flags);

	dev_info(&odev->pdev->dev, "Queue %u reset complete\n", qid);
	return 0;
}
