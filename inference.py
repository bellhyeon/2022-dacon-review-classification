import random
import re
import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    AdamW,
    get_cosine_schedule_with_warmup,
)


def seed_everything(seed: int = 42) -> None:
    print(f"Global Seed Set to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clean(review: str) -> str:
    BMP_pattern = re.compile(
        "[" "\U00010000-\U0010FFFF" "]+", flags=re.UNICODE  # BMP characters 이외
    )

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    review = re.sub("([ㄱ-ㅎ ㅏ-ㅣ]+)", " ", review)  # 단일 자음, 모음 제거
    review = re.sub(
        "[-=+,#/\?:^$.@*& ​\^\"※ᆢㆍ;~♡!』<>.,‘|\(\)\[\]`'…》\”\“\’·]", " ", review
    )  # 특수문자 제거

    review = BMP_pattern.sub(r"", review)
    review = emoji_pattern.sub(r"", review)
    review = " ".join(review.split())
    return review


class RatingDataset(Dataset):
    def __init__(self, reviews, ratings, args, is_training):
        self.is_training = is_training
        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
        self.max_length = args["max_length"]
        self.reviews = reviews
        self.ratings = ratings

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        if self.is_training:
            assert self.ratings is not None
            reviews = clean(self.reviews[item])
            labels = self.ratings[item]
            inputs = self.tokenizer(
                reviews,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                add_special_tokens=True,
            )
            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "token_type_ids": inputs["token_type_ids"][0],
                "label": torch.tensor(labels, dtype=torch.long),
            }
        else:
            assert self.ratings is None
            reviews = clean(self.reviews[item])
            inputs = self.tokenizer(
                reviews,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                add_special_tokens=True,
            )
            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "token_type_ids": inputs["token_type_ids"][0],
            }


class RatingModel(nn.Module):
    def __init__(self):
        super(RatingModel, self).__init__()
        self.config = AutoConfig.from_pretrained(args["model_name"])
        self.config.classifier_dropout = args["dropout_rate"]
        self.config.num_labels = 4
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args["model_name"], config=self.config
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids, attention_mask, token_type_ids)
        return output


def kfold_inference(args, test_reviews, model_saved_path):
    test_dataset = RatingDataset(
        test_reviews, ratings=None, args=args, is_training=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
    )

    model = RatingModel()

    optimizer = AdamW(model.parameters(), lr=args["lr"])

    total_steps = int(len(test_dataloader) * args["epochs"] / args["batch_size"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * total_steps),
        num_training_steps=total_steps,
    )
    checkpoint = torch.load(model_saved_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    prediction = []

    model.eval()
    with torch.no_grad():
        for item in tqdm(test_dataloader):
            input_ids = item["input_ids"].cuda()
            attention_masks = item["attention_mask"].cuda()
            token_type_ids = item["token_type_ids"].cuda()
            outputs = model(input_ids, attention_masks, token_type_ids)
            preds = outputs.logits
            prediction.extend(preds.data.cpu().numpy())

    return np.array(prediction)


if __name__ == "__main__":

    args = {
        "save_path": "./models/saved_model/",
        "model_saved_path": "./models/saved_model/1/*/model.pt",
        "seed": 42,
        "model_name": "kykim/electra-kor-base",
        "batch_size": 32,
        "lr": 1e-5,
        "weight_decay": 0.0,
        "epochs": 10,
        "max_length": 128,
        "train_data_path": "dataset/train.csv",
        "test_data_path": "dataset/test.csv",
        "num_workers": 4,
        "device": 0,
        "dropout_rate": 0.3,
        "warmup_rate": 0.1,
        "label_smoothing": 0.1,
        "num_folds": 5,
        "total_steps": 195,
        "warmup_steps": 19,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_everything(args["seed"])

    test_df = pd.read_csv(args["test_data_path"])
    submission = pd.read_csv("dataset/sample_submission.csv")
    test_reviews = test_df["reviews"].tolist()

    infer_results = []

    save_model_paths = sorted(glob(args["model_saved_path"]))
    print(save_model_paths)

    for fold_num, save_model_path in enumerate(save_model_paths):
        print("=" * 100)
        print(f"Model trained fold : {fold_num + 1}")
        print(f"Saved Model path : {save_model_path}")
        infer_result = kfold_inference(args, test_reviews, save_model_path)
        infer_results.append(infer_result)

        fold_prediction = [np.argmax(i) for i in infer_result]
        submission["target"] = fold_prediction
        submission.to_csv(f"{args['save_path']}/fold_{fold_num + 1}.csv", index=False)

    prediction = (
        infer_results[0]
        + infer_results[1]
        + infer_results[2]
        + infer_results[3]
        + infer_results[4]
    )

    prediction = prediction / 5

    prediction = [np.argmax(i) for i in prediction]

    print("Done.")

    submission["target"] = prediction
    label_encode_dict = {0: 1, 1: 2, 2: 4, 3: 5}
    submission = submission.replace({"target": label_encode_dict})

    submission.to_csv(f"{args['save_path']}/submission_soft.csv", index=False)

    print("Hard Voting")
    df0 = pd.read_csv(f"{args['save_path']}/fold_1.csv")
    df1 = pd.read_csv(f"{args['save_path']}/fold_2.csv")
    df2 = pd.read_csv(f"{args['save_path']}/fold_3.csv")
    df3 = pd.read_csv(f"{args['save_path']}/fold_4.csv")
    df4 = pd.read_csv(f"{args['save_path']}/fold_5.csv")

    df = pd.DataFrame(
        {
            "fold_1": df0.target,
            "fold_2": df1.target,
            "fold_3": df2.target,
            "fold_4": df3.target,
            "fold_5": df4.target,
        }
    )

    df.to_csv(f"{args['save_path']}/hard_voting.csv", index=False)

    df["target"] = df.mode(axis=1)[0]

    submission["target"] = df.target.astype(int).tolist()

    submission = submission.replace({"target": label_encode_dict})
    submission.to_csv(f"{args['save_path']}/submission_hard.csv", index=False)
