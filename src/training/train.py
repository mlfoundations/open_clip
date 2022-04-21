import logging
import math
import time

from .train_jig import TrainJig
from .utils import AverageMeter

try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(jig: TrainJig, data, scheduler, args, tb_writer=None):
    dev_env = jig.dev_env
    epoch = jig.start_train_epoch()

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, (images, texts) in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images = images.to(device=dev_env.device, non_blocking=True)
        texts = texts.to(device=dev_env.device, non_blocking=True)
        data_time_m.update(time.time() - end)

        output = jig.train_step(images, texts)

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if dev_env.is_master and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            total_loss = output['total_loss'].item()
            loss_m.update(total_loss, batch_size)
            logit_scale = output['logit_scale'].item()
            current_lr = jig.get_current_lr()

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {current_lr:5f} "
                f"Logit Scale: {logit_scale:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "scale":  logit_scale,
                "lr": current_lr,
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

    # end for
    jig.end_train_epoch()


