from hydra.utils import instantiate

def build_model(cfg):
    return instantiate(cfg.model.net)

def build_scheduler(cfg):
    return instantiate(cfg.scheduler.scheduler)

def build_optimizer(cfg, model):
    return instantiate(cfg.optimizer, params=model.parameters())