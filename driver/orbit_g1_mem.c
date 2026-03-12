// SPDX-License-Identifier: GPL-2.0
/*
 * ORBIT-G1 PCIe Accelerator Driver
 * orbit_g1_mem.c — GDDR6 buddy allocator and BAR1 mmap handler
 */
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/errno.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/io.h>
#include <linux/pci.h>

#include "orbit_g1.h"

/* =========================================================================
 * Buddy Allocator — init / fini
 * ========================================================================= */

/**
 * orbit_g1_mem_init - Initialise the GDDR6 buddy allocator.
 *
 * The full GDDR6 address space (as reported by the device at probe time)
 * is handed to the allocator as a single free block at max_order.
 *
 * Min allocation granularity: 4 KB (order 12).
 * Max allocation order: determined from the actual GDDR6 size at runtime.
 */
int orbit_g1_mem_init(struct orbit_g1_device *odev)
{
	struct orbit_mem_pool *pool = &odev->gddr_pool;
	struct orbit_mem_block *block;
	u64 size = odev->gddr_size_bytes;
	u32 max_order;
	int i;

	if (size == 0) {
		dev_warn(&odev->pdev->dev,
			 "GDDR6 size is 0 — memory pool disabled\n");
		return 0;
	}

	/* Compute max order: largest power-of-two that fits within size */
	max_order = ORBIT_MEM_MIN_ORDER;
	while (max_order < ORBIT_MEM_MAX_ORDER &&
	       (1ULL << (max_order + 1)) <= size)
		max_order++;

	pool->base      = 0;		/* GDDR6 address space starts at 0 */
	pool->size      = size;
	pool->min_order = ORBIT_MEM_MIN_ORDER;
	pool->max_order = max_order;
	pool->allocated_bytes = 0;
	spin_lock_init(&pool->lock);

	for (i = 0; i < ORBIT_MEM_NUM_ORDERS; i++)
		INIT_LIST_HEAD(&pool->free_lists[i]);

	INIT_LIST_HEAD(&pool->alloc_list);

	/*
	 * Insert the full GDDR6 range as a single free block at max_order.
	 *
	 * If the GDDR6 size is not a power-of-two, the remainder is handled
	 * by inserting additional blocks that cover the residual range using
	 * a binary decomposition (similar to the Linux bootmem free path).
	 *
	 * TODO: implement full binary decomposition for non-power-of-two sizes.
	 * For now, insert one block at max_order (may waste up to 50% for
	 * non-power-of-two sizes — acceptable for a skeleton).
	 */
	block = kzalloc(sizeof(*block), GFP_KERNEL);
	if (!block)
		return -ENOMEM;

	block->base  = pool->base;
	block->order = max_order;
	INIT_LIST_HEAD(&block->list);
	list_add(&block->list, &pool->free_lists[max_order - pool->min_order]);

	dev_info(&odev->pdev->dev,
		 "GDDR6 pool: base=0x%llx size=%llu MB min_order=%u max_order=%u\n",
		 pool->base, pool->size >> 20, pool->min_order, pool->max_order);

	return 0;
}

/**
 * orbit_g1_mem_fini - Free all memory pool metadata.
 */
void orbit_g1_mem_fini(struct orbit_g1_device *odev)
{
	struct orbit_mem_pool *pool = &odev->gddr_pool;
	struct orbit_mem_block *block, *tmp;
	int i;
	unsigned long flags;

	spin_lock_irqsave(&pool->lock, flags);

	for (i = 0; i < ORBIT_MEM_NUM_ORDERS; i++) {
		list_for_each_entry_safe(block, tmp, &pool->free_lists[i], list) {
			list_del(&block->list);
			kfree(block);
		}
	}

	/* Free any blocks still on alloc_list (leaked allocations). */
	list_for_each_entry_safe(block, tmp, &pool->alloc_list, list) {
		list_del(&block->list);
		kfree(block);
	}

	spin_unlock_irqrestore(&pool->lock, flags);
}

/* =========================================================================
 * Buddy Allocator — alloc / free
 * ========================================================================= */

/**
 * order_for_size - Return the minimum buddy order to satisfy @size_bytes.
 */
static u32 order_for_size(struct orbit_mem_pool *pool, u64 size_bytes)
{
	u32 order = pool->min_order;

	while (order <= pool->max_order && (1ULL << order) < size_bytes)
		order++;

	return order;
}

/**
 * orbit_mem_alloc - Allocate a GDDR6 region from the buddy pool.
 *
 * @pool:            the GDDR6 memory pool
 * @size_bytes:      requested size (will be rounded up to power-of-two)
 * @align_bytes:     required alignment (must be power-of-two, >= 4096)
 * @device_addr_out: filled with the GDDR6-physical base address on success
 * @handle_out:      filled with an opaque token (== device_addr) for free
 *
 * Returns 0 on success, -ENOMEM if pool is exhausted, -EINVAL for bad args.
 */
int orbit_mem_alloc(struct orbit_mem_pool *pool,
		    u64 size_bytes, u64 align_bytes,
		    u64 *device_addr_out, u64 *handle_out)
{
	struct orbit_mem_block *block, *tmp;
	unsigned long flags;
	u32 req_order, list_idx, split_order;

	if (!pool || !device_addr_out || !handle_out)
		return -EINVAL;
	if (size_bytes == 0)
		return -EINVAL;
	if (align_bytes < (1ULL << pool->min_order))
		align_bytes = (1ULL << pool->min_order);

	req_order = order_for_size(pool, size_bytes);
	/* Alignment may require a higher order if align > size */
	while (req_order <= pool->max_order &&
	       (1ULL << req_order) < align_bytes)
		req_order++;

	if (req_order > pool->max_order)
		return -EINVAL;	/* requested more than max allocation */

	spin_lock_irqsave(&pool->lock, flags);

	/*
	 * Find the smallest free block at or above req_order.
	 * Then split down to req_order, returning buddies to lower free lists.
	 */
	for (list_idx = req_order - pool->min_order;
	     list_idx < ORBIT_MEM_NUM_ORDERS;
	     list_idx++) {
		if (!list_empty(&pool->free_lists[list_idx])) {
			block = list_first_entry(&pool->free_lists[list_idx],
						 struct orbit_mem_block, list);
			list_del(&block->list);
			goto found;
		}
	}

	spin_unlock_irqrestore(&pool->lock, flags);
	return -ENOMEM;

found:
	split_order = pool->min_order + list_idx;

	/* Split the block down to req_order, inserting buddies into free lists */
	while (split_order > req_order) {
		struct orbit_mem_block *buddy;

		split_order--;

		buddy = kzalloc(sizeof(*buddy), GFP_ATOMIC);
		if (!buddy) {
			/*
			 * TODO: handle allocation failure more gracefully.
			 * For now, put the original block back and fail.
			 */
			list_add(&block->list,
				 &pool->free_lists[split_order + 1 - pool->min_order]);
			spin_unlock_irqrestore(&pool->lock, flags);
			return -ENOMEM;
		}

		/* Buddy starts at block->base + 2^split_order */
		buddy->base  = block->base + (1ULL << split_order);
		buddy->order = split_order;
		INIT_LIST_HEAD(&buddy->list);

		list_add(&buddy->list,
			 &pool->free_lists[split_order - pool->min_order]);
	}

	block->order = req_order;
	pool->allocated_bytes += (1ULL << req_order);

	/*
	 * Track the allocated block in pool->alloc_list so orbit_mem_free()
	 * can recover the real order without a radix tree.
	 * block->list is reused as the alloc_list node (it was removed from
	 * the free list above, so the node is free to re-link here).
	 * Insert while still holding the lock so alloc_list is always
	 * consistent under pool->lock.
	 */
	INIT_LIST_HEAD(&block->list);
	list_add(&block->list, &pool->alloc_list);

	spin_unlock_irqrestore(&pool->lock, flags);

	*device_addr_out = block->base;
	*handle_out      = block->base;	/* handle == device_addr (unique) */

	dev_dbg(NULL, "orbit_mem_alloc: addr=0x%llx order=%u\n",
		*device_addr_out, req_order);

	return 0;
}

/**
 * orbit_mem_free - Return a GDDR6 region to the buddy pool.
 *
 * @pool:   the GDDR6 memory pool
 * @handle: the handle returned by orbit_mem_alloc()
 *
 * Walks pool->alloc_list to find the block with matching dev_addr so the
 * real allocation order is recovered.  Without this, buddy-free would always
 * use min_order, silently under-returning memory and corrupting the pool.
 *
 * TODO: implement full buddy coalescing (merge with free buddies).
 */
int orbit_mem_free(struct orbit_mem_pool *pool, u64 handle)
{
	struct orbit_mem_block *block, *found = NULL;
	unsigned long flags;

	if (!pool || handle == 0)
		return -EINVAL;

	spin_lock_irqsave(&pool->lock, flags);

	/*
	 * Walk alloc_list to locate the block by device address (== handle).
	 * This gives us the correct order for buddy-free.
	 */
	list_for_each_entry(block, &pool->alloc_list, list) {
		if (block->base == handle) {
			found = block;
			list_del(&found->list);
			break;
		}
	}

	if (!found) {
		spin_unlock_irqrestore(&pool->lock, flags);
		return -ENOENT;
	}

	/*
	 * Return the block to the appropriate free list using its real order.
	 * TODO: implement full buddy coalescing loop.
	 */
	INIT_LIST_HEAD(&found->list);
	list_add(&found->list,
		 &pool->free_lists[found->order - pool->min_order]);

	if (pool->allocated_bytes >= (1ULL << found->order))
		pool->allocated_bytes -= (1ULL << found->order);

	spin_unlock_irqrestore(&pool->lock, flags);

	return 0;
}

/* =========================================================================
 * BAR1 mmap handler — zero-copy weight upload
 * ========================================================================= */

/**
 * orbit_g1_mmap - mmap file operation for /dev/orbit_g1_N.
 *
 * Maps a portion of BAR1 (GDDR6 window) into the calling process's address
 * space using write-combining page protection.
 *
 * vma->vm_pgoff encodes the byte offset within BAR1 (shifted by PAGE_SHIFT
 * as per the mmap syscall convention).
 */
int orbit_g1_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct orbit_fd_ctx *ctx = filp->private_data;
	struct orbit_g1_device *odev = ctx->dev;
	struct pci_dev *pdev = odev->pdev;
	unsigned long offset, size;
	unsigned long pfn;
	int ret;

	if (!odev->bar1) {
		dev_err(&pdev->dev, "mmap: BAR1 not available\n");
		return -ENXIO;
	}

	offset = vma->vm_pgoff << PAGE_SHIFT;
	size   = vma->vm_end - vma->vm_start;

	/* Bounds check */
	if (offset >= odev->bar1_len) {
		dev_err(&pdev->dev,
			"mmap: offset 0x%lx >= bar1_len 0x%llx\n",
			offset, (u64)odev->bar1_len);
		return -EINVAL;
	}
	if (size > odev->bar1_len - offset) {
		dev_err(&pdev->dev,
			"mmap: size 0x%lx exceeds BAR1 window\n", size);
		return -EINVAL;
	}

	/* Apply write-combining page protection */
	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

	/* Mark the VMA as I/O (not backed by normal memory) */
	vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;

	pfn = (pci_resource_start(pdev, ORBIT_BAR1) + offset) >> PAGE_SHIFT;

	ret = io_remap_pfn_range(vma,
				 vma->vm_start,
				 pfn,
				 size,
				 vma->vm_page_prot);
	if (ret) {
		dev_err(&pdev->dev, "io_remap_pfn_range failed: %d\n", ret);
		return ret;
	}

	dev_dbg(&pdev->dev,
		"mmap BAR1: offset=0x%lx size=0x%lx pfn=0x%lx\n",
		offset, size, pfn);

	return 0;
}
