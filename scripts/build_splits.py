import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

MAIN_DIR = Path("/ix/rbao/shared/rbao_qug14_ngl18/cd8_manifests/patch_512")
TRAIN_SAMP = 2000
VAL_TEST_SAMP = 1000
RAND_STATE = 42

def main():
    case_nums = range(1, 83)
    
    train, test = train_test_split(case_nums, test_size=0.3, random_state=RAND_STATE)
    valid, test = train_test_split(test, test_size=0.5, random_state=RAND_STATE)

    print("Building train set")
    train_samps = []
    for case in train:
        records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
        samp_size = min(TRAIN_SAMP, len(records))
        sample = records.sample(samp_size, random_state=RAND_STATE)
        train_samps.append(sample)

    train_df = pd.concat(train_samps, ignore_index=True)
    train_df.to_csv(MAIN_DIR / "cd8_train_manifest.csv", index=False)

    print("Building validation set")
    valid_samps = []
    for case in valid:
        records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
        samp_size = min(VAL_TEST_SAMP, len(records))
        sample = records.sample(samp_size, random_state=RAND_STATE)
        valid_samps.append(sample)

    valid_df = pd.concat(valid_samps, ignore_index=True)
    valid_df.to_csv(MAIN_DIR / "cd8_valid_manifest.csv", index=False)

    print("Building test set")
    test_samps = []
    for case in test:
        records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
        samp_size = min(VAL_TEST_SAMP, len(records))
        sample = records.sample(samp_size, random_state=RAND_STATE)
        test_samps.append(sample)

    test_df = pd.concat(test_samps, ignore_index=True)
    test_df.to_csv(MAIN_DIR / "cd8_test_manifest.csv", index=False)

    print("Splits Created")
    print(f"Train Dataset: {len(train)} WSIs, {len(train_df)} tiles")
    print(f"Validation Dataset: {len(valid)} WSIs, {len(valid_df)} tiles")
    print(f"Test Dataset: {len(test)} WSIs, {len(test_df)} tiles")

    # print("Train")
    # for case in train:
    #     records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
    #     num_records = len(records)
    #     print(f"T{str(case).zfill(3)}: {num_records} tiles")
    
    # print("Validation")
    # for case in valid:
    #     records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
    #     num_records = len(records)
    #     print(f"T{str(case).zfill(3)}: {num_records} tiles")

    # print("Test")
    # for case in test:
    #     records = pd.read_csv(MAIN_DIR / f"T{str(case).zfill(3)}_tiles.csv")
    #     num_records = len(records)
    #     print(f"T{str(case).zfill(3)}: {num_records} tiles")


    
if __name__ == "__main__":
    main()