import matplotlib.pyplot as plt
import math
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import torchmetrics
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import torch.utils.data as data
from torchvision import transforms
import ast
from torch.nn.utils.rnn import pad_sequence

users = pd.read_csv(
    'user.csv',engine='python'
)

actions = pd.read_csv(
    'action.csv',engine='python'
)
cats = pd.read_csv(
    'category.csv',engine='python'
)

class Userbehavior(data.Dataset):
    def __init__(
        self, actions_file,test=False
    ):
        self.actions_frame = pd.read_csv(
            actions_file,
            delimiter=",",

        )
        self.test = test

    def __len__(self):
        return len(self.actions_frame)

    def __getitem__(self, idx):
        data = self.actions_frame.iloc[idx]
        user_id = data.user_id

        cat_all = eval(data.sequence_cat_id)
        cat_all_actions = eval(data.sequence_action_type)
        target_cat_id = cat_all[-1:][0]
        target_cat_actions = cat_all_actions[-1:][0]

        cat_all = torch.LongTensor(cat_all[:-1])
        cat_all_actions = torch.LongTensor(cat_all_actions[:-1])

        age_range = data.age_range
        occupation = data.occupation
        location=data.location

        return user_id, cat_all, target_cat_id, cat_all_actions, target_cat_actions, age_range, occupation,\
               location
styles = [
        "Home",
        "Shoes",
        "Cartoon",
        "Baby",
        "Bags",
        "Medias",
        "Clothing",
        "Accessories",
        "Sports",
        "Cellphone",
        "Toys",
        "Networks",
        "Gardening",
        "PersonalCare",
        "Grocery",
        "Health",
        "Cars",
        "Beauty",
    ]


sequence_length = 6

class BST(pl.LightningModule):
    def __init__(
            self, args=None,
    ):
        super().__init__()
        super(BST, self).__init__()

        self.save_hyperparameters()
        self.args = args

        # Embedding layers
        ##Users
        self.embeddings_user_id = nn.Embedding(
            int(users.user_id.max()) + 1, int(math.sqrt(users.user_id.max())) + 1
        )
        ###Users features embeddings
        self.embeddings_user_age = nn.Embedding(
            len(users.age.unique()), int(math.sqrt(len(users.age.unique())))
        )
        self.embeddings_user_occupation = nn.Embedding(
            len(users.occupation.unique()), int(math.sqrt(len(users.occupation.unique())))
        )

        ##Item
        self.embeddings_cat_id = nn.Embedding(
            int(cats.cat_id.max()) + 1, int(math.sqrt(cats.cat_id.max())) + 1
        )
        self.embeddings_position = nn.Embedding(
            sequence_length, int(math.sqrt(len(cats.cat_id.unique()))) + 1
        )
        #Item features embeddings
        style_vectors = cats[styles].to_numpy()
        self.embeddings_cat_style = nn.Embedding(
            style_vectors.shape[0], style_vectors.shape[1]
        )

        self.embeddings_cat_shoplevel = nn.Embedding(
            len(cats.shoplevel.unique()), int(math.sqrt(len(cats.shoplevel.unique())))
        )

        self.transfomerlayer = nn.TransformerEncoderLayer(63, 3, dropout=0.2)
        self.linear = nn.Sequential(
            nn.Linear(
                334,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.auc=torchmetrics.AUC(reorder=True)
        self.acc=torchmetrics.Accuracy()


    def encode_input(self, inputs):
        user_id, cat_all, target_cat_id, cat_all_actions, target_cat_action, gender, age_,occupation= inputs
        cat_all = self.embeddings_cat_id(cat_all)
        target_cat = self.embeddings_cat_id(target_cat_id)

        positions = torch.arange(0, sequence_length - 1, 1, dtype=int, device=self.device)
        positions = self.embeddings_position(positions)

        encoded_sequence_cat_with_poistion_and_actions = (cat_all + positions)  # Yet to multiply by rating

        target_cat = torch.unsqueeze(target_cat, 1)
        transfomer_features = torch.cat((encoded_sequence_cat_with_poistion_and_actions, target_cat), dim=1)

        # USERS
        user_id = self.embeddings_user_id(user_id)

        age= self.embeddings_user_age(age)
        occupation = self.embeddings_user_occupation(occupation)

        user_features = torch.cat((user_id, age, occupation), 1)

        return transfomer_features, user_features, target_cat_action.float()

    def forward(self, batch):
        transfomer_features, user_features, target_cat_action = self.encode_input(batch)
        first_output = self.transfomerlayer(transfomer_features)
        first_output = torch.flatten(first_output, start_dim=1)


        features = torch.cat((first_output, user_features), dim=1)

        output = self.linear(features)
        return output, target_cat_action

    def training_step(self, batch, batch_idx):
        out, target_cat_action = self(batch)
        out = out.flatten()
        loss = self.criterion(out, target_cat_action)

        mae = self.mae(out, target_cat_action)
        mse = self.mse(out, target_cat_action)
        rmse = torch.sqrt(mse)
        self.log(
            "train/mse", mae, on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
             "train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False
        )
        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0005)

    def add_model_specific_args(parent_parser):
        parser =argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("rate", type=float, default=0.01)
        return parser

    def setup(self, stage=None):
        print("start to load")
        self.train_dataset = Userbehavior("train.csv")
        self.test_dataset = Userbehavior('test.csv')
        print("Model setup is done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

