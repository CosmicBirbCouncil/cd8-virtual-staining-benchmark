import sys
from trident.patch_encoder_models.load import encoder_factory as patch_encoder_model_factory

def main():
    trident_path = "/ix3/cpace/ngl18/projects/TRIDENT"

    if trident_path not in sys.path:
        sys.path.insert(0, repo_str)

    patch_encoder_models = [
        "hoptimus1",  "hibou_l", "conch_v15"
    ]
    for model in patch_encoder_models:
        try:
            patch_encoder_model_factory(model)
        except Exception as e:
            print(f"Failed to download weights for {model}: {e}")

if __name__ == "__main__":
    main()