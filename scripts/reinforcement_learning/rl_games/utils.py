def set_terminations(env, cfg):
    """
    Removes active termination terms from the environment according
    to specified terms in cfg.

    param cfg: dict {term: active} containing term name to active bool mappings
    """
    manager = env.unwrapped.termination_manager
    for term, active in cfg.items():
        if not active:
            try:
                idx = manager._term_names.index(term)
                manager._term_names.pop(idx)
                manager._term_cfgs.pop(idx)
                manager._term_dones.pop(term)
            except ValueError:
                continue
