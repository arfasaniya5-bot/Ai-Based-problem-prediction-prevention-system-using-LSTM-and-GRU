"""Model Inference Service for DeviceGuard AI.

Loads the verified LSTM neural network checkpoint and preprocessor configuration.
Performs strictly consistent feature ordering, imputation, RobustScaling, and forward pass.
Never fits a new scaler or imputer at inference time.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import settings
from src.models.architectures import LSTMClassifier
from src.schemas.prediction import DailyTelemetryInput, PredictionResponse

logger = logging.getLogger(__name__)


class ModelInferenceService:
    """Encapsulates model loading and inference lifecycle."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.device = torch.device("cpu")
        self.lstm_model: Optional[LSTMClassifier] = None
        self.config: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.imputer_medians: Dict[str, float] = {}
        self.scaler_center: Dict[str, float] = {}
        self.scaler_scale: Dict[str, float] = {}
        self.threshold: float = settings.DECISION_THRESHOLD
        self.model_version: str = settings.APP_VERSION

        self.models_status: Dict[str, bool] = {
            "lstm": False,
            "gru": False,
            "baseline": False,
        }

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load preprocessor configuration and neural network checkpoints."""
        config_path = self.models_dir / "preprocessor_config.json"
        if not config_path.exists():
            logger.error(f"Preprocessor config not found at: {config_path}")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            self.feature_names = self.config.get("feature_names", [])
            self.imputer_medians = self.config.get("imputer_medians", {})
            self.scaler_center = self.config.get("scaler_center", {})
            self.scaler_scale = self.config.get("scaler_scale", {})
            logger.info(f"Loaded preprocessor configuration for {len(self.feature_names)} features.")
        except Exception as e:
            logger.error(f"Failed to parse preprocessor config: {e}")
            return

        # Load LSTM Checkpoint
        lstm_checkpoint_path = self.models_dir / "best_lstm.pt"
        if lstm_checkpoint_path.exists():
            try:
                model = LSTMClassifier(
                    input_size=len(self.feature_names),
                    hidden_size=32,
                    num_layers=1,
                    dropout=0.2,
                )
                model.load_state_dict(torch.load(lstm_checkpoint_path, map_location=self.device, weights_only=True))
                model.to(self.device)
                model.eval()
                self.lstm_model = model
                self.models_status["lstm"] = True
                logger.info(f"Loaded LSTM checkpoint from {lstm_checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed loading LSTM model: {e}")
        else:
            logger.warning(f"LSTM checkpoint not found at {lstm_checkpoint_path}")

        # Check GRU Checkpoint existence
        gru_path = self.models_dir / "best_gru.pt"
        self.models_status["gru"] = gru_path.exists()

        # Check Baseline existence
        baseline_path = self.models_dir / "baseline_model.joblib"
        self.models_status["baseline"] = baseline_path.exists()

    def get_health_status(self) -> Dict[str, Any]:
        """Return diagnostic health information for model serving."""
        all_ready = self.lstm_model is not None and bool(self.feature_names)
        return {
            "status": "loaded" if all_ready else "degraded",
            "models_loaded": self.models_status,
            "feature_count": len(self.feature_names),
            "sequence_length": settings.SEQUENCE_LENGTH,
            "horizon_days": settings.HORIZON_DAYS,
            "decision_threshold": self.threshold,
            "device": str(self.device),
            "timestamp": datetime.now(timezone.utc),
        }

    def preprocess_sequence(self, readings: List[DailyTelemetryInput]) -> np.ndarray:
        """Apply strict feature ordering, training median imputation, and RobustScaling.

        Args:
            readings: List of exactly 3 DailyTelemetryInput objects.

        Returns:
            Numpy array of shape (3, num_features).
        """
        if len(readings) != settings.SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected exactly {settings.SEQUENCE_LENGTH} consecutive daily readings, got {len(readings)}"
            )

        processed_timesteps = []
        for step_idx, reading in enumerate(readings):
            reading_dict = reading.model_dump()
            step_values = []

            for feature in self.feature_names:
                raw_val = reading_dict.get(feature)

                # Missing value handling consistent with training:
                # If feature value is None, impute using verified training median
                if raw_val is None:
                    imputed_val = self.imputer_medians.get(feature, 0.0)
                else:
                    imputed_val = float(raw_val)

                # Apply RobustScaler: (x - center) / scale
                center = self.scaler_center.get(feature, 0.0)
                scale = self.scaler_scale.get(feature, 1.0)
                if scale == 0.0 or np.isnan(scale):
                    scaled_val = 0.0
                else:
                    scaled_val = (imputed_val - center) / scale

                step_values.append(scaled_val)

            processed_timesteps.append(step_values)

        return np.array(processed_timesteps, dtype=np.float32)

    def predict(
        self,
        readings: List[DailyTelemetryInput],
        serial_number: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> PredictionResponse:
        """Execute model forward pass and format prediction response."""
        if self.lstm_model is None:
            raise RuntimeError("LSTM model artifact is not loaded or unavailable")

        # Preprocess features
        seq_array = self.preprocess_sequence(readings)  # Shape: (3, 11)
        input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, 3, 11)

        with torch.no_grad():
            logits = self.lstm_model(input_tensor)
            risk_score = float(torch.sigmoid(logits).item())

        # Determine predicted class based on verified threshold
        predicted_class = 1 if risk_score >= self.threshold else 0
        is_failure_predicted = bool(predicted_class == 1)

        # Categorize risk level
        if risk_score >= self.threshold:
            risk_level = "CRITICAL"
            recommendation = (
                "Critical failure risk detected within 48 hours. "
                "Initiate immediate automated data backup and schedule storage unit replacement."
            )
        elif risk_score >= 0.25:
            risk_level = "MODERATE"
            recommendation = (
                "Moderate wear telemetry observed. "
                "Increase SMART monitoring frequency and ensure preventative data redundancy."
            )
        else:
            risk_level = "LOW"
            recommendation = "Drive operating within nominal parameters. No maintenance intervention required."

        return PredictionResponse(
            serial_number=serial_number,
            model_name=settings.DEFAULT_MODEL_NAME,
            model_version=self.model_version,
            prediction_class=predicted_class,
            is_failure_predicted=is_failure_predicted,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            threshold=self.threshold,
            horizon_days=settings.HORIZON_DAYS,
            sequence_length_days=settings.SEQUENCE_LENGTH,
            recommendation=recommendation,
            timestamp=datetime.now(timezone.utc),
        )


# Global singleton instance
_inference_service: Optional[ModelInferenceService] = None


def get_inference_service() -> ModelInferenceService:
    """Retrieve or initialize the global ModelInferenceService singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = ModelInferenceService()
    return _inference_service
