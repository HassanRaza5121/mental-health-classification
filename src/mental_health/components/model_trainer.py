import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam

from src.mental_health.logger import logger
from src.mental_health.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.train_data_path = "artifacts/transformed_train_dataset.csv"
        self.model_dir = "artifacts/models"

    def train_classical_models(self, X, y):

        models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=100),
            "svm": SVC(probability=True),
            "xgboost": XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                num_class=len(np.unique(y))
            )
        }

        saved_models = {}

        for name, model in models.items():
            logger.info(f"Training {name}")

            model.fit(X, y)

            model_path = os.path.join(self.model_dir, f"{name}.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"{name} saved at {model_path}")
            saved_models[name] = model_path

        return saved_models

    def train_1dcnn(self, X, y, epochs=20, batch_size=32):

        X_cnn = X.values.reshape((X.shape[0], X.shape[1], 1))

        num_classes = len(np.unique(y))

        model = Sequential([
            Conv1D(32, kernel_size=2, activation="relu", input_shape=(X.shape[1], 1)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(X_cnn, y.values, epochs=epochs, batch_size=batch_size, verbose=0)

        model_path = os.path.join(self.model_dir, "1dcnn_model.h5")
        model.save(model_path)

        logger.info(f"1D-CNN saved at {model_path}")

        return {"1dcnn": model_path}

    def train_all_models(self):

        try:
            logger.info("Starting Model Training")

            train_df = pd.read_csv(self.train_data_path)
            target_column = "stress_experience"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            # ✅ Ensure correct label format
            y_train = y_train.astype(int)

            # ✅ Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            os.makedirs(self.model_dir, exist_ok=True)

            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            logger.info(f"Scaler saved at {scaler_path}")

            X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)

            # ✅ Train models
            classical_models = self.train_classical_models(X_scaled_df, y_train)
            cnn_model = self.train_1dcnn(X_scaled_df, y_train)

            all_models = {**classical_models, **cnn_model}

            logger.info("All models trained successfully")

            return all_models

        except Exception as e:
            raise CustomException(e, sys)