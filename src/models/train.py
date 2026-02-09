"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       learning_rate: float = 0.01,
                       epochs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a linear model using gradient descent.

    Args:
        X_train: Training data (N, T, H, W, C)
        y_train: Training labels (N,)
        X_val: Validation data (opcional)
        y_val: Validation labels (opcional)
        learning_rate: Learning rate
        epochs: Number of seasons

    Returns:
        Tuple with trained weights and biases
    """
    logger.info(
        f"Starting training: {X_train.shape[0]} samples, {epochs} eras")

    # Flatten the input data
    N_train = X_train.shape[0]
    X_train_flat = X_train.reshape(N_train, -1)

    # Initializing parameters
    input_size = X_train_flat.shape[1]
    weights = np.random.randn(input_size) * 0.01
    bias = np.random.randn(1) * 0.01

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predictions = X_train_flat @ weights + bias

        # Calculates error (MSE)
        errors = predictions - y_train
        loss = np.mean(errors ** 2)

        # Backward pass (gradientes)
        grad_weights = (2 / N_train) * (X_train_flat.T @ errors)
        grad_bias = (2 / N_train) * np.sum(errors)

        # Updating parameters
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        # Periodic Log
        if (epoch + 1) % 10 == 0:
            logger.info(f"Season {epoch + 1}/{epochs}, Loss: {loss:.6f}")

            # Validation if provided
            if X_val is not None and y_val is not None:
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                val_predictions = X_val_flat @ weights + bias
                val_loss = np.mean((val_predictions - y_val) ** 2)
                logger.info(f"  Val Loss: {val_loss:.6f}")

    logger.info(f"Training completed. Final loss.: {loss:.6f}")
    return weights, bias
