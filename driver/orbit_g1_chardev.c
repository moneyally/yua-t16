// SPDX-License-Identifier: GPL-2.0
/*
 * ORBIT-G1 PCIe Accelerator Driver
 * orbit_g1_chardev.c — /dev/orbit_g1_N character device + ioctl dispatch
 */
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/atomic.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/errno.h>
#include <linux/poll.h>
#include <linux/mm.h>

#include "orbit_g1.h"

/* =========================================================================
 * Helpers: per-device minor counter
 * ========================================================================= */
static atomic_t orbit_minor_counter = ATOMIC_INIT(0);

/* =========================================================================
 * File operations: open / release
 * ========================================================================= */

static int orbit_g1_open(struct inode *inode, struct file *filp)
{
	struct orbit_g1_device *odev =
		container_of(inode->i_cdev, struct orbit_g1_device, cdev);
	struct orbit_fd_ctx *ctx;
	static atomic_t session_counter = ATOMIC_INIT(0);

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;

	ctx->dev        = odev;
	ctx->session_id = (u32)atomic_inc_return(&session_counter);
	spin_lock_init(&ctx->alloc_lock);
	INIT_LIST_HEAD(&ctx->alloc_list);

	filp->private_data = ctx;
	atomic_inc(&odev->open_count);

	dev_dbg(&odev->pdev->dev,
		"open: session_id=%u open_count=%d\n",
		ctx->session_id, atomic_read(&odev->open_count));

	return 0;
}

static int orbit_g1_release(struct inode *inode, struct file *filp)
{
	struct orbit_fd_ctx *ctx = filp->private_data;
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_alloc_entry *entry, *tmp;
	unsigned long flags;

	/* Free all GDDR6 allocations owned by this fd */
	spin_lock_irqsave(&ctx->alloc_lock, flags);
	list_for_each_entry_safe(entry, tmp, &ctx->alloc_list, list) {
		list_del(&entry->list);
		spin_unlock_irqrestore(&ctx->alloc_lock, flags);

		orbit_mem_free(&odev->gddr_pool, entry->handle);
		kfree(entry);

		spin_lock_irqsave(&ctx->alloc_lock, flags);
	}
	spin_unlock_irqrestore(&ctx->alloc_lock, flags);

	atomic_dec(&odev->open_count);
	filp->private_data = NULL;
	kfree(ctx);

	return 0;
}

/* =========================================================================
 * poll (non-blocking completion wait)
 * ========================================================================= */

static __poll_t orbit_g1_poll(struct file *filp, poll_table *wait)
{
	struct orbit_fd_ctx *ctx = filp->private_data;
	struct orbit_g1_device *odev = ctx->dev;
	__poll_t mask = 0;
	int i;

	/* Add all queue wait queues to the poll table */
	for (i = 0; i < ORBIT_NUM_QUEUES; i++)
		poll_wait(filp, &odev->queues[i].wq, wait);

	/* Signal POLLIN if any queue has no inflight work */
	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		if (atomic_read(&odev->queues[i].inflight) == 0) {
			mask |= EPOLLIN | EPOLLRDNORM;
			break;
		}
	}

	return mask;
}

/* =========================================================================
 * ioctl handlers
 * ========================================================================= */

/* --- ORBIT_IOC_SUBMIT_DESC --- */
static long ioctl_submit_desc(struct orbit_fd_ctx *ctx,
			       struct orbit_desc_submit __user *uarg)
{
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_desc_submit req;
	void *kdescs;
	size_t total_bytes;
	u64 cookie = 0;
	int ret;

	if (copy_from_user(&req, uarg, sizeof(req)))
		return -EFAULT;

	if (req.queue_id >= ORBIT_NUM_QUEUES)
		return -EINVAL;
	if (req.count == 0 || req.count > ORBIT_QUEUE_DEPTH)
		return -EINVAL;
	if (!req.descs_ptr)
		return -EINVAL;

	total_bytes = (size_t)req.count * ORBIT_DESC_SIZE;

	kdescs = kmalloc(total_bytes, GFP_KERNEL);
	if (!kdescs)
		return -ENOMEM;

	if (copy_from_user(kdescs, (void __user *)(uintptr_t)req.descs_ptr,
			   total_bytes)) {
		kfree(kdescs);
		return -EFAULT;
	}

	/*
	 * TODO: validate each descriptor:
	 *   - type field in known range (0x01–0x0F)
	 *   - address fields 64-byte aligned
	 *   - no reserved fields set
	 */

	ret = orbit_queue_submit(&odev->queues[req.queue_id],
				 kdescs, req.count, &cookie);
	kfree(kdescs);
	if (ret)
		return ret;

	/* Write back the cookie */
	if (put_user(cookie, &uarg->submit_cookie))
		return -EFAULT;

	/* If ORBIT_SUBMIT_WAIT: block until this batch completes */
	if (req.flags & ORBIT_SUBMIT_WAIT) {
		ret = orbit_queue_wait(&odev->queues[req.queue_id],
				       cookie, 5000 /* 5 s default */);
		if (ret)
			return ret;
	}

	return 0;
}

/* --- ORBIT_IOC_WAIT_DONE --- */
static long ioctl_wait_done(struct orbit_fd_ctx *ctx,
			     struct orbit_wait_done __user *uarg)
{
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_wait_done req;

	if (copy_from_user(&req, uarg, sizeof(req)))
		return -EFAULT;

	if (req.queue_id >= ORBIT_NUM_QUEUES)
		return -EINVAL;

	return orbit_queue_wait(&odev->queues[req.queue_id],
				req.submit_cookie,
				req.timeout_ms);
}

/* --- ORBIT_IOC_ALLOC_MEM --- */
static long ioctl_alloc_mem(struct orbit_fd_ctx *ctx,
			     struct orbit_mem_alloc __user *uarg)
{
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_alloc_entry *entry;
	struct orbit_mem_alloc req;
	u64 device_addr, handle;
	unsigned long flags;
	int ret;

	if (copy_from_user(&req, uarg, sizeof(req)))
		return -EFAULT;

	if (req.size_bytes == 0)
		return -EINVAL;
	if (req.align_bytes == 0)
		req.align_bytes = PAGE_SIZE;
	/* align must be power-of-two */
	if (req.align_bytes & (req.align_bytes - 1))
		return -EINVAL;

	ret = orbit_mem_alloc(&odev->gddr_pool,
			      req.size_bytes,
			      req.align_bytes,
			      &device_addr,
			      &handle);
	if (ret)
		return ret;

	/* Track the allocation in the per-fd list for cleanup on release() */
	entry = kzalloc(sizeof(*entry), GFP_KERNEL);
	if (!entry) {
		orbit_mem_free(&odev->gddr_pool, handle);
		return -ENOMEM;
	}

	entry->handle      = handle;
	entry->device_addr = device_addr;
	entry->size_bytes  = req.size_bytes;
	INIT_LIST_HEAD(&entry->list);

	spin_lock_irqsave(&ctx->alloc_lock, flags);
	list_add(&entry->list, &ctx->alloc_list);
	spin_unlock_irqrestore(&ctx->alloc_lock, flags);

	/* Write back results */
	if (put_user(device_addr, &uarg->device_addr) ||
	    put_user(handle,      &uarg->handle)) {
		/* Clean up if write-back fails */
		spin_lock_irqsave(&ctx->alloc_lock, flags);
		list_del(&entry->list);
		spin_unlock_irqrestore(&ctx->alloc_lock, flags);
		orbit_mem_free(&odev->gddr_pool, handle);
		kfree(entry);
		return -EFAULT;
	}

	return 0;
}

/* --- ORBIT_IOC_FREE_MEM --- */
static long ioctl_free_mem(struct orbit_fd_ctx *ctx,
			    struct orbit_mem_free __user *uarg)
{
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_alloc_entry *entry, *found = NULL;
	struct orbit_mem_free req;
	unsigned long flags;

	if (copy_from_user(&req, uarg, sizeof(req)))
		return -EFAULT;

	if (req.handle == 0)
		return -EINVAL;

	/* Find and remove from per-fd list */
	spin_lock_irqsave(&ctx->alloc_lock, flags);
	list_for_each_entry(entry, &ctx->alloc_list, list) {
		if (entry->handle == req.handle) {
			found = entry;
			list_del(&found->list);
			break;
		}
	}
	spin_unlock_irqrestore(&ctx->alloc_lock, flags);

	if (!found)
		return -ENOENT;

	orbit_mem_free(&odev->gddr_pool, found->handle);
	kfree(found);

	return 0;
}

/* --- ORBIT_IOC_GET_INFO --- */
static long ioctl_get_info(struct orbit_fd_ctx *ctx,
			    struct orbit_device_info __user *uarg)
{
	struct orbit_g1_device *odev = ctx->dev;
	struct orbit_device_info info;

	memset(&info, 0, sizeof(info));

	info.gddr_size_bytes  = odev->gddr_size_bytes;
	info.bar1_size_bytes  = (u64)odev->bar1_len;
	info.num_queues       = ORBIT_NUM_QUEUES;
	info.queue_depth      = ORBIT_QUEUE_DEPTH;
	info.fw_version       = odev->fw_version;
	info.hw_revision      = odev->hw_revision;
	info.desc_spec_version = 2;	/* v2 extensions supported */
	/*
	 * Bitmask of supported descriptor type IDs 1-15 (types 0x01-0x0F).
	 * Bit N corresponds to type ID N.  Bit 0 (type 0) is reserved/unused.
	 * 0xFFFE = bits 1..15 set = types 1..15 supported.
	 */
	info.supported_desc_types = 0xFFFEU;

	strscpy(info.device_name, "ORBIT-G1", sizeof(info.device_name));

	if (copy_to_user(uarg, &info, sizeof(info)))
		return -EFAULT;

	return 0;
}

/* --- ORBIT_IOC_RESET_QUEUE --- */
static long ioctl_reset_queue(struct orbit_fd_ctx *ctx,
			       struct orbit_queue_reset __user *uarg)
{
	struct orbit_queue_reset req;

	if (copy_from_user(&req, uarg, sizeof(req)))
		return -EFAULT;

	if (req.queue_id >= ORBIT_NUM_QUEUES)
		return -EINVAL;
	if (req.flags != 0)
		return -EINVAL;

	return orbit_queue_reset(ctx->dev, req.queue_id);
}

/* =========================================================================
 * ioctl dispatch
 * ========================================================================= */

static long orbit_g1_ioctl(struct file *filp, unsigned int cmd,
			    unsigned long arg)
{
	struct orbit_fd_ctx *ctx = filp->private_data;
	void __user *uarg = (void __user *)arg;

	if (!ctx)
		return -ENODEV;

	switch (cmd) {
	case ORBIT_IOC_SUBMIT_DESC:
		return ioctl_submit_desc(ctx, uarg);
	case ORBIT_IOC_WAIT_DONE:
		return ioctl_wait_done(ctx, uarg);
	case ORBIT_IOC_ALLOC_MEM:
		return ioctl_alloc_mem(ctx, uarg);
	case ORBIT_IOC_FREE_MEM:
		return ioctl_free_mem(ctx, uarg);
	case ORBIT_IOC_GET_INFO:
		return ioctl_get_info(ctx, uarg);
	case ORBIT_IOC_RESET_QUEUE:
		return ioctl_reset_queue(ctx, uarg);
	default:
		return -ENOTTY;
	}
}

/* =========================================================================
 * file_operations table
 * ========================================================================= */

/* Forward declaration: orbit_g1_mmap is defined in orbit_g1_mem.c */
int orbit_g1_mmap(struct file *filp, struct vm_area_struct *vma);

static const struct file_operations orbit_g1_fops = {
	.owner          = THIS_MODULE,
	.open           = orbit_g1_open,
	.release        = orbit_g1_release,
	.unlocked_ioctl = orbit_g1_ioctl,
	.mmap           = orbit_g1_mmap,
	.poll           = orbit_g1_poll,
	.llseek         = no_llseek,
};

/* =========================================================================
 * cdev create / destroy
 * ========================================================================= */

/**
 * orbit_g1_cdev_create - Register the character device for @odev.
 *
 * Creates /dev/orbit_g1_N and the corresponding /sys/class entry.
 */
int orbit_g1_cdev_create(struct orbit_g1_device *odev)
{
	int minor;
	dev_t devt;
	int ret;

	minor = atomic_inc_return(&orbit_minor_counter) - 1;
	devt  = MKDEV(MAJOR(orbit_g1_devt_base), minor);

	odev->minor = minor;
	odev->devt  = devt;

	cdev_init(&odev->cdev, &orbit_g1_fops);
	odev->cdev.owner = THIS_MODULE;

	ret = cdev_add(&odev->cdev, devt, 1);
	if (ret) {
		dev_err(&odev->pdev->dev, "cdev_add failed: %d\n", ret);
		return ret;
	}

	odev->dev = device_create(orbit_g1_class,
				  &odev->pdev->dev,
				  devt,
				  odev,
				  "orbit_g1_%d", minor);
	if (IS_ERR(odev->dev)) {
		ret = PTR_ERR(odev->dev);
		dev_err(&odev->pdev->dev,
			"device_create failed: %d\n", ret);
		cdev_del(&odev->cdev);
		return ret;
	}

	dev_info(&odev->pdev->dev,
		 "Created /dev/orbit_g1_%d (major=%u minor=%u)\n",
		 minor, MAJOR(devt), minor);

	return 0;
}

/**
 * orbit_g1_cdev_destroy - Unregister the character device for @odev.
 */
void orbit_g1_cdev_destroy(struct orbit_g1_device *odev)
{
	if (odev->dev) {
		device_destroy(orbit_g1_class, odev->devt);
		odev->dev = NULL;
	}
	cdev_del(&odev->cdev);

	dev_info(&odev->pdev->dev, "Removed /dev/orbit_g1_%d\n", odev->minor);
}
