"""Backend-independent registration for future post-CFG compatibility nodes."""


POST_CFG_OPTION = "sampler_post_cfg_function"


def register_post_cfg_callback(model, callback):
    """Clone a ModelPatcher and append a callback without sharing its list."""
    if not callable(callback):
        raise TypeError("post-CFG callback must be callable.")
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("model must expose ModelPatcher-compatible clone().")
    branch = clone()
    if not isinstance(getattr(branch, "model_options", None), dict):
        raise TypeError("cloned model must expose dict model_options.")
    callbacks = branch.model_options.get(POST_CFG_OPTION, ())
    if not isinstance(callbacks, (list, tuple)):
        raise TypeError(f"{POST_CFG_OPTION} must be a list or tuple.")
    branch.model_options[POST_CFG_OPTION] = [*callbacks, callback]
    return branch


__all__ = ["POST_CFG_OPTION", "register_post_cfg_callback"]
