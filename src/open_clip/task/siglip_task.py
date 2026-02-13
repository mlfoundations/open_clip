from .clip_task import CLIPTask


class SigLIPTask(CLIPTask):
    """SigLIP task. Structurally identical to CLIPTask (loss handles differences).

    Separate class for type distinction and future SigLIP-specific behavior.
    """
    pass
