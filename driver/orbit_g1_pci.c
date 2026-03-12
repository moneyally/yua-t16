// SPDX-License-Identifier: GPL-2.0
/*
 * ORBIT-G1 PCIe Accelerator Driver
 * orbit_g1_pci.c — PCI probe/remove, BAR mapping, MSI-X init, module init/exit
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/interrupt.h>

#include "orbit_g1.h"

MODULE_AUTHOR("ORBIT-G1 Driver Authors");
MODULE_DESCRIPTION("ORBIT-G1 PCIe AI Accelerator Driver");
MODULE_LICENSE("GPL v2");
MODULE_VERSION("0.1.0");

/* =========================================================================
 * Module-wide globals
 * ========================================================================= */
struct class *orbit_g1_class;
EXPORT_SYMBOL_GPL(orbit_g1_class);

dev_t orbit_g1_devt_base;
EXPORT_SYMBOL_GPL(orbit_g1_devt_base);

#define ORBIT_G1_MAX_DEVS	16

/* =========================================================================
 * IRQ handlers — one per queue
 * ========================================================================= */
irqreturn_t orbit_irq_common(struct orbit_g1_device *dev, int qid)
{
	/* Clear the per-queue interrupt bit (RW1C) */
	orbit_write32(dev, ORBIT_REG_INTR_STATUS, BIT(qid));
	/* Advance completion tracking and wake up waiters */
	orbit_queue_complete(dev, qid);
	return IRQ_HANDLED;
}

irqreturn_t orbit_irq_q0(int irq, void *data)
{
	return orbit_irq_common((struct orbit_g1_device *)data, 0);
}

irqreturn_t orbit_irq_q1(int irq, void *data)
{
	return orbit_irq_common((struct orbit_g1_device *)data, 1);
}

irqreturn_t orbit_irq_q2(int irq, void *data)
{
	return orbit_irq_common((struct orbit_g1_device *)data, 2);
}

irqreturn_t orbit_irq_q3(int irq, void *data)
{
	return orbit_irq_common((struct orbit_g1_device *)data, 3);
}

typedef irqreturn_t (*orbit_irq_fn_t)(int, void *);
static const orbit_irq_fn_t orbit_irq_handlers[ORBIT_NUM_QUEUES] = {
	orbit_irq_q0,
	orbit_irq_q1,
	orbit_irq_q2,
	orbit_irq_q3,
};

/* =========================================================================
 * MMIO init/fini
 * ========================================================================= */

/**
 * orbit_g1_mmio_init - Map BAR0 (control registers) and BAR1 (GDDR6 window).
 *
 * BAR0 is mapped with ioremap() (strongly ordered for register access).
 * BAR1 is mapped with ioremap_wc() (write-combining for DMA throughput).
 */
int orbit_g1_mmio_init(struct orbit_g1_device *odev)
{
	struct pci_dev *pdev = odev->pdev;
	u32 id_reg;

	/* --- BAR0 --- */
	odev->bar0_len = pci_resource_len(pdev, ORBIT_BAR0);
	if (odev->bar0_len == 0) {
		dev_err(&pdev->dev, "BAR0 has zero length\n");
		return -ENOMEM;
	}

	odev->bar0 = pci_ioremap_bar(pdev, ORBIT_BAR0);
	if (!odev->bar0) {
		dev_err(&pdev->dev, "Failed to ioremap BAR0\n");
		return -ENOMEM;
	}

	/* Verify device magic register */
	id_reg = orbit_read32(odev, ORBIT_REG_ID);
	if (id_reg != ORBIT_REG_ID_MAGIC) {
		dev_warn(&pdev->dev,
			 "BAR0 ID mismatch: expected 0x%08X, got 0x%08X\n",
			 ORBIT_REG_ID_MAGIC, id_reg);
		/* Non-fatal in skeleton; real driver may return -ENODEV */
	}

	/* Read static device info */
	odev->hw_revision    = orbit_read32(odev, ORBIT_REG_HW_REV);
	odev->fw_version     = orbit_read32(odev, ORBIT_REG_FW_VER);
	odev->gddr_size_bytes = orbit_read64(odev, ORBIT_REG_GDDR_SIZE_LO);

	dev_info(&pdev->dev, "BAR0 mapped: 0x%llx len=0x%llx\n",
		 (u64)pci_resource_start(pdev, ORBIT_BAR0),
		 (u64)odev->bar0_len);
	dev_info(&pdev->dev, "HW rev=0x%08x FW ver=0x%08x GDDR6=%llu MB\n",
		 odev->hw_revision, odev->fw_version,
		 odev->gddr_size_bytes >> 20);

	/* --- BAR1 --- */
	odev->bar1_len = pci_resource_len(pdev, ORBIT_BAR1);
	if (odev->bar1_len == 0) {
		dev_warn(&pdev->dev, "BAR1 not present or zero length — mmap disabled\n");
		odev->bar1 = NULL;
		return 0;
	}

	odev->bar1 = ioremap_wc(pci_resource_start(pdev, ORBIT_BAR1),
				 odev->bar1_len);
	if (!odev->bar1) {
		dev_err(&pdev->dev, "Failed to ioremap_wc BAR1\n");
		iounmap(odev->bar0);
		odev->bar0 = NULL;
		return -ENOMEM;
	}

	dev_info(&pdev->dev, "BAR1 mapped (WC): 0x%llx len=0x%llx\n",
		 (u64)pci_resource_start(pdev, ORBIT_BAR1),
		 (u64)odev->bar1_len);

	return 0;
}

void orbit_g1_mmio_fini(struct orbit_g1_device *odev)
{
	if (odev->bar1) {
		iounmap(odev->bar1);
		odev->bar1 = NULL;
	}
	if (odev->bar0) {
		iounmap(odev->bar0);
		odev->bar0 = NULL;
	}
}

/* =========================================================================
 * MSI-X init/fini
 * ========================================================================= */

static int orbit_g1_irq_init(struct orbit_g1_device *odev)
{
	struct pci_dev *pdev = odev->pdev;
	int nvec, i, ret;
	char irq_name[32];

	nvec = pci_alloc_irq_vectors(pdev,
				     ORBIT_NUM_QUEUES,
				     ORBIT_NUM_QUEUES,
				     PCI_IRQ_MSIX);
	if (nvec < 0) {
		dev_err(&pdev->dev,
			"Failed to allocate %d MSI-X vectors: %d\n",
			ORBIT_NUM_QUEUES, nvec);
		return nvec;
	}

	odev->msix_nvec = nvec;
	dev_info(&pdev->dev, "Allocated %d MSI-X vectors\n", nvec);

	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		snprintf(irq_name, sizeof(irq_name), "orbit_g1_q%d", i);
		ret = request_irq(pci_irq_vector(pdev, i),
				  orbit_irq_handlers[i],
				  0,
				  irq_name,
				  odev);
		if (ret) {
			dev_err(&pdev->dev,
				"request_irq q%d failed: %d\n", i, ret);
			goto err_free_irqs;
		}

		/* Tell the hardware which MSI-X vector to use for this queue */
		orbit_write32(odev, ORBIT_Q_INTR_VECTOR(i), i);
	}

	/* Unmask all queue interrupts */
	orbit_write32(odev, ORBIT_REG_INTR_MASK, 0xF);

	return 0;

err_free_irqs:
	while (--i >= 0)
		free_irq(pci_irq_vector(pdev, i), odev);
	pci_free_irq_vectors(pdev);
	return ret;
}

static void orbit_g1_irq_fini(struct orbit_g1_device *odev)
{
	struct pci_dev *pdev = odev->pdev;
	int i;

	/* Mask all interrupts */
	orbit_write32(odev, ORBIT_REG_INTR_MASK, 0x0);

	for (i = 0; i < odev->msix_nvec; i++)
		free_irq(pci_irq_vector(pdev, i), odev);

	pci_free_irq_vectors(pdev);
	odev->msix_nvec = 0;
}

/* =========================================================================
 * DMA ring allocation helpers (descriptor ring coherent buffers)
 * ========================================================================= */

static int orbit_g1_dma_init(struct orbit_g1_device *odev)
{
	struct pci_dev *pdev = odev->pdev;
	size_t ring_bytes = ORBIT_QUEUE_DEPTH * ORBIT_DESC_SIZE;
	int i;

	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		struct orbit_queue *q = &odev->queues[i];

		q->ring_cpu = dma_alloc_coherent(&pdev->dev,
						 ring_bytes,
						 &q->ring_dma,
						 GFP_KERNEL);
		if (!q->ring_cpu) {
			dev_err(&pdev->dev,
				"dma_alloc_coherent failed for queue %d\n", i);
			goto err_free_rings;
		}

		memset(q->ring_cpu, 0, ring_bytes);
		q->ring_size = ORBIT_QUEUE_DEPTH;

		dev_dbg(&pdev->dev,
			"Queue %d ring: cpu=%p dma=0x%llx size=%zu\n",
			i, q->ring_cpu, (u64)q->ring_dma, ring_bytes);
	}

	return 0;

err_free_rings:
	while (--i >= 0) {
		struct orbit_queue *q = &odev->queues[i];

		if (q->ring_cpu) {
			dma_free_coherent(&pdev->dev, ring_bytes,
					  q->ring_cpu, q->ring_dma);
			q->ring_cpu = NULL;
		}
	}
	return -ENOMEM;
}

static void orbit_g1_dma_fini(struct orbit_g1_device *odev)
{
	size_t ring_bytes = ORBIT_QUEUE_DEPTH * ORBIT_DESC_SIZE;
	int i;

	for (i = 0; i < ORBIT_NUM_QUEUES; i++) {
		struct orbit_queue *q = &odev->queues[i];

		if (q->ring_cpu) {
			dma_free_coherent(&odev->pdev->dev, ring_bytes,
					  q->ring_cpu, q->ring_dma);
			q->ring_cpu = NULL;
		}
	}
}

/* =========================================================================
 * PCI probe
 * ========================================================================= */

static int orbit_g1_probe(struct pci_dev *pdev,
			   const struct pci_device_id *id)
{
	struct orbit_g1_device *odev;
	int ret;

	dev_info(&pdev->dev, "ORBIT-G1 probe: vendor=0x%04x device=0x%04x\n",
		 pdev->vendor, pdev->device);

	/* Allocate device private data */
	odev = kzalloc(sizeof(*odev), GFP_KERNEL);
	if (!odev)
		return -ENOMEM;

	odev->pdev = pdev;
	spin_lock_init(&odev->lock);
	atomic_set(&odev->open_count, 0);

	/* 1. Enable PCI device */
	ret = pci_enable_device(pdev);
	if (ret) {
		dev_err(&pdev->dev, "pci_enable_device failed: %d\n", ret);
		goto err_free;
	}

	/* 2. Enable bus mastering for DMA */
	pci_set_master(pdev);

	/* 3. Configure DMA mask (prefer 64-bit, fall back to 32-bit) */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
		if (ret) {
			dev_err(&pdev->dev, "Cannot set DMA mask\n");
			goto err_disable;
		}
	}

	/* 4. Reserve BAR0 and BAR1 */
	ret = pci_request_regions(pdev, "orbit_g1");
	if (ret) {
		dev_err(&pdev->dev, "pci_request_regions failed: %d\n", ret);
		goto err_disable;
	}

	/* 5. Map BAR0/BAR1 */
	ret = orbit_g1_mmio_init(odev);
	if (ret)
		goto err_release_regions;

	/* 6. Allocate DMA coherent descriptor rings */
	ret = orbit_g1_dma_init(odev);
	if (ret)
		goto err_mmio_fini;

	/* 7. Initialize queue ring buffer state */
	ret = orbit_g1_queue_init(odev);
	if (ret)
		goto err_dma_fini;

	/* 8. Set up MSI-X interrupts */
	ret = orbit_g1_irq_init(odev);
	if (ret)
		goto err_queue_fini;

	/* 9. Initialize GDDR6 buddy allocator */
	ret = orbit_g1_mem_init(odev);
	if (ret)
		goto err_irq_fini;

	/* 10. Create character device /dev/orbit_g1_N */
	ret = orbit_g1_cdev_create(odev);
	if (ret)
		goto err_mem_fini;

	/* 11. Store driver private data */
	pci_set_drvdata(pdev, odev);

	dev_info(&pdev->dev, "ORBIT-G1 ready\n");
	return 0;

err_mem_fini:
	orbit_g1_mem_fini(odev);
err_irq_fini:
	orbit_g1_irq_fini(odev);
err_queue_fini:
	orbit_g1_queue_fini(odev);
err_dma_fini:
	orbit_g1_dma_fini(odev);
err_mmio_fini:
	orbit_g1_mmio_fini(odev);
err_release_regions:
	pci_release_regions(pdev);
err_disable:
	pci_disable_device(pdev);
err_free:
	kfree(odev);
	return ret;
}

/* =========================================================================
 * PCI remove
 * ========================================================================= */

static void orbit_g1_remove(struct pci_dev *pdev)
{
	struct orbit_g1_device *odev = pci_get_drvdata(pdev);

	if (!odev)
		return;

	dev_info(&pdev->dev, "ORBIT-G1 remove\n");

	/* Reverse order of probe */
	orbit_g1_cdev_destroy(odev);
	orbit_g1_mem_fini(odev);
	orbit_g1_irq_fini(odev);
	orbit_g1_queue_fini(odev);
	orbit_g1_dma_fini(odev);
	orbit_g1_mmio_fini(odev);
	pci_release_regions(pdev);
	pci_disable_device(pdev);
	pci_set_drvdata(pdev, NULL);
	kfree(odev);
}

/* =========================================================================
 * PCI device ID table
 * ========================================================================= */

static const struct pci_device_id orbit_g1_pci_ids[] = {
	{ PCI_DEVICE(ORBIT_G1_VENDOR_ID, ORBIT_G1_DEVICE_ID) },
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, orbit_g1_pci_ids);

static struct pci_driver orbit_g1_pci_driver = {
	.name		= "orbit_g1",
	.id_table	= orbit_g1_pci_ids,
	.probe		= orbit_g1_probe,
	.remove		= orbit_g1_remove,
};

/* =========================================================================
 * Module init / exit
 * ========================================================================= */

static int __init orbit_g1_init(void)
{
	int ret;

	/* Allocate a range of device numbers for up to ORBIT_G1_MAX_DEVS */
	ret = alloc_chrdev_region(&orbit_g1_devt_base, 0,
				  ORBIT_G1_MAX_DEVS, "orbit_g1");
	if (ret) {
		pr_err("orbit_g1: alloc_chrdev_region failed: %d\n", ret);
		return ret;
	}

	/* Create /sys/class/orbit_g1 */
	orbit_g1_class = class_create("orbit_g1");
	if (IS_ERR(orbit_g1_class)) {
		ret = PTR_ERR(orbit_g1_class);
		pr_err("orbit_g1: class_create failed: %d\n", ret);
		goto err_chrdev;
	}

	ret = pci_register_driver(&orbit_g1_pci_driver);
	if (ret) {
		pr_err("orbit_g1: pci_register_driver failed: %d\n", ret);
		goto err_class;
	}

	pr_info("orbit_g1: module loaded\n");
	return 0;

err_class:
	class_destroy(orbit_g1_class);
err_chrdev:
	unregister_chrdev_region(orbit_g1_devt_base, ORBIT_G1_MAX_DEVS);
	return ret;
}

static void __exit orbit_g1_exit(void)
{
	pci_unregister_driver(&orbit_g1_pci_driver);
	class_destroy(orbit_g1_class);
	unregister_chrdev_region(orbit_g1_devt_base, ORBIT_G1_MAX_DEVS);
	pr_info("orbit_g1: module unloaded\n");
}

module_init(orbit_g1_init);
module_exit(orbit_g1_exit);
