"""Translate legacy training settings into the standalone loss factory's explicit options."""
from open_clip.factory import create_loss, _use_loss_label_cache
from open_clip.model_traits import ModelFamily, get_model_traits, validate_distillation
from open_clip.task import unwrap_model


def create_loss_from_args(args, model):
    """Build the legacy trainer's loss using built-model traits and its raw-label padding convention.

    The task-based trainer constructs its own losses; in particular CoCaTask masks labels to -100
    and uses pad_id=None. This adapter instead reads model.pad_id for legacy raw caption labels.
    """
    model = unwrap_model(model)
    traits = get_model_traits(model)
    if args.distill:
        validate_distillation(traits)
        loss_type = "distill_clip"
    elif traits.family in (ModelFamily.COCA, ModelFamily.MAMMUT):
        loss_type = "coca"
    elif traits.family in (ModelFamily.GENLIP, ModelFamily.GENLAP):
        loss_type = "genlip"
    else:
        loss_type = "siglip" if args.siglip else "clip"

    kwargs = {}
    if loss_type != "genlip":
        kwargs.update(rank=args.rank, world_size=args.world_size, cache_labels=_use_loss_label_cache(args))
    if loss_type in ("clip", "distill_clip", "coca"):
        kwargs.update(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad)
    if loss_type in ("coca", "genlip"):
        kwargs.update(
            z_loss_weight=getattr(args, 'caption_z_loss_weight', 0.0),
            compute_dtype=getattr(args, 'caption_loss_compute_dtype', 'float32'),
        )
    if loss_type == "coca":
        kwargs.update(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            pad_id=model.pad_id,
        )
    elif loss_type == "siglip":
        kwargs['dist_impl'] = args.loss_dist_impl

    return create_loss(loss_type, **kwargs)
