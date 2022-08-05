import json
import random
import shutil
import re
import os
from typing import List, Dict
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

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


def get_save_kfold_model_path(save_path: str, save_model_name: str, fold_num: int):
    save_folder_path = os.path.join(save_path, str(fold_num + 1))
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path


def save_model(save_path, model, optimizer, scheduler):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        save_path,
    )


def calc_accuracy(preds, label):
    _, max_indices = torch.max(preds, 1)
    accuracy = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]
    return accuracy


def save_loss_graph(save_folder_path: str, train_loss: List, valid_loss: List):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "loss.png")
    plt.savefig(save_path)


def save_acc_graph(
    save_folder_path: str, train_acc: List, valid_acc: List,
):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_acc")
    plt.plot(valid_acc, label="valid_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("ACC", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "acc.png")
    plt.savefig(save_path)


def kfold_main_loop(
    args, train_reviews, train_ratings, valid_reviews, valid_ratings, fold_num
):
    train_dataset = RatingDataset(train_reviews, train_ratings, args, is_training=True)
    valid_dataset = RatingDataset(valid_reviews, valid_ratings, args, is_training=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
    )

    save_model_path, save_folder_path = get_save_kfold_model_path(
        args["save_path"], "model.pt", fold_num
    )

    model = RatingModel()
    model.cuda()

    optimizer = AdamW(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    total_steps = int(len(train_dataloader) * args["epochs"] / args["batch_size"])
    warmup_steps = int(args["warmup_rate"] * total_steps)
    print(f"total_steps: {total_steps}, num_warmup_steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])

    prev_valid_acc = 1e-4
    t_loss, t_acc = [], []
    v_loss, v_acc = [], []
    for epoch in range(args["epochs"]):
        print("=" * 25 + f"Epoch {epoch + 1} Train" + "=" * 25)
        total_train_loss = 0.0
        total_train_acc = 0.0

        total_valid_loss = 0.0
        total_valid_acc = 0.0

        model.train()

        for item in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = item["input_ids"].cuda()
            attention_masks = item["attention_mask"].cuda()
            token_type_ids = item["token_type_ids"].cuda()
            labels = item["label"].cuda()
            outputs = model(input_ids, attention_masks, token_type_ids)
            preds = outputs.logits
            loss = criterion(preds, labels)
            total_train_loss += loss.data
            total_train_acc += calc_accuracy(preds, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        t_loss.append(avg_train_loss.item())
        t_acc.append(avg_train_acc)
        print(f"Train loss : {avg_train_loss:.5f}, Train acc : {avg_train_acc:.5f}")

        print("=" * 25 + f"Epoch {epoch + 1} Valid" + "=" * 25)
        model.eval()
        with torch.no_grad():
            for item in tqdm(valid_dataloader):
                input_ids = item["input_ids"].cuda()
                attention_masks = item["attention_mask"].cuda()
                token_type_ids = item["token_type_ids"].cuda()
                labels = item["label"].cuda()
                outputs = model(input_ids, attention_masks, token_type_ids)
                preds = outputs.logits
                loss = criterion(preds, labels)
                total_valid_loss += loss.data
                total_valid_acc += calc_accuracy(preds, labels)
            avg_valid_loss = total_valid_loss / len(valid_dataloader)
            avg_valid_acc = total_valid_acc / len(valid_dataloader)
            v_loss.append(avg_valid_loss.item())
            v_acc.append(avg_valid_acc)
            print(
                f"Valid loss: {avg_valid_loss:.5f}, Valid acc: {avg_valid_acc:.5f}, BEST: {prev_valid_acc:.5f}, lr: {optimizer.param_groups[0]['lr']}"
            )
            save_loss_graph(save_folder_path, t_loss, v_loss)
            save_acc_graph(save_folder_path, t_acc, v_acc)

            if prev_valid_acc < avg_valid_acc:
                prev_valid_acc = avg_valid_acc
                save_model(save_model_path, model, optimizer, scheduler)

                save_result_path = os.path.join(save_folder_path, "result.json")
                with open(save_result_path, "w") as json_file:
                    save_result_dict: Dict = {
                        "best_epoch": epoch + 1,
                        "train_loss": round(avg_train_loss.item(), 4),
                        "valid_loss": round(avg_valid_loss.item(), 4),
                        "train_acc": f"{avg_train_acc:.5f}",
                        "valid_acc": f"{avg_valid_acc:.5f}",
                    }

                    json.dump(save_result_dict, json_file)
                print("Save Model and Graph\n")


if __name__ == "__main__":

    args = {
        "save_path": "./models/saved_model/",
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["device"])
    seed_everything(args["seed"])

    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])

    num_folder = len(glob(args["save_path"] + "*"))
    args["save_path"] = os.path.join(args["save_path"], str(num_folder + 1))

    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])
    shutil.copyfile("./train.py", os.path.join(args["save_path"], "train.py"))

    label_encode_dict = {1: 0, 2: 1, 4: 2, 5: 3}

    train_df = pd.read_csv(args["train_data_path"])
    train_df = train_df.replace({"target": label_encode_dict})
    reviews = train_df["reviews"].tolist()
    ratings = train_df["target"].tolist()

    fold_list = []
    skf = StratifiedKFold(
        n_splits=args["num_folds"], shuffle=True, random_state=args["seed"]
    )
    for train, valid in skf.split(reviews, ratings):
        fold_list.append([train, valid])
        print("train", len(train), train)
        print("valid", len(valid), valid)
        print()

    for fold_num, fold in enumerate(fold_list):
        print(f"Fold num : {str(fold_num + 1)}, fold : {fold}")
        train_reviews = [reviews[i] for i in fold[0]]
        train_ratings = [ratings[i] for i in fold[0]]

        valid_reviews = [reviews[i] for i in fold[1]]
        valid_ratings = [ratings[i] for i in fold[1]]

        kfold_main_loop(
            args, train_reviews, train_ratings, valid_reviews, valid_ratings, fold_num
        )
