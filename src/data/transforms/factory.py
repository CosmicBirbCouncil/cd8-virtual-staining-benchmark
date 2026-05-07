from hydra.utils import instantiate

def build_transform(cfg, split: str):
    if split == "train":
        return instantiate(cfg.transform.train)
    if split == "val":
        return instantiate(cfg.transform.val)
    if split == "test":
        return instantiate(cfg.transform.test)
    raise ValueError(f"Unsupported split: {split}")