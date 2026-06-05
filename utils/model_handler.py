import logging
from pathlib import Path

import xgboost as xgb

logger = logging.getLogger(__name__)


def save_xgb_model(model: xgb.Booster, model_name: str, model_dir: str) -> None:
    file_dir = Path(model_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{model_name}.json"

    model_path = file_dir / model_name

    model.save_model(model_path)

    logger.info(f"Model saved to {model_path}")
