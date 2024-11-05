import os
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader

from utils import *
from dataset import TimeSeriesDataset
from models.transformer import TransformerClassifier
from trainer import Trainer

def main(ratio, scaler_type="standard"):
    seed_everything(43)

    # Load configuration
    hyperparams = load_config("transformer")

    # Unpack hyperparameters
    threshold = hyperparams["threshold"]
    window_minutes = hyperparams["window_minutes"]
    stride_minutes = hyperparams["stride_minutes"]
    length = hyperparams["length"]
    batch_size = hyperparams["batch_size"]
    dim = hyperparams["dim"]
    num_heads = hyperparams["num_heads"]
    hidden_dim = hyperparams["hidden_dim"]
    num_layers = hyperparams["num_layers"]
    learning_rate = hyperparams["learning_rate"]
    epochs = hyperparams["epochs"]

    # Read data
    full_path = os.getcwd()
    data_path = os.path.join(full_path, 'data', '경진대회용 주조 공정최적화 데이터셋.csv')
    data = pd.read_csv(data_path, encoding='cp949')

    # Preprocess
    data = preprocess(data)
    data_time_series = make_time_series(data, time_threshold=3000)  # 3000초 => 50분
    data_time_series = preprocess_time_series(data_time_series)

    # Split
    train, valid, test = split_by_process(data_time_series)
    train, valid, test = interpolate(train, valid, test)

    # Apply scaler
    scaler = apply_scaler(train, scaler_type=scaler_type)

    train_dataset = TimeSeriesDataset(train, scaler=scaler, threshold=threshold, window_minutes=window_minutes, stride_minutes=stride_minutes, length=length, undersampling=True, ratio=ratio)
    valid_dataset = TimeSeriesDataset(valid, scaler=scaler, threshold=threshold, window_minutes=window_minutes, stride_minutes=stride_minutes, length=length)
    test_dataset = TimeSeriesDataset(test, scaler=scaler, threshold=threshold, window_minutes=window_minutes, stride_minutes=stride_minutes, length=length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss, trainer
    model = TransformerClassifier(input_dim=dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=cross_entropy_loss, save_model_path='model_saved_dl/transformer.pth')

    # Train
    trainer.fit(train_loader, valid_loader, epochs=epochs)

    # Infer
    trainer.evaluate_metrics(test_loader, model_path='model_saved_dl/transformer.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Transformer model.")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio for undersampling.")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"], help="Scaler type to use (default: standard).")

    args = parser.parse_args()
    
    main(ratio=args.ratio, scaler_type=args.scaler)
