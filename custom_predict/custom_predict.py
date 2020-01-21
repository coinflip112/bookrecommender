import os
import joblib
from pathlib import Path
import numpy as np
import tensorflow as tf


class CustomPredict(object):
    def __init__(self, model, item_encoder, user_encoder, to_mask_items_mapping):
        self._model = model
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.to_mask_items_mapping = to_mask_items_mapping

    def predict(self, instances, **kwargs):
        user_id = instances[0]
        mapped_user_id = self.user_encoder.transform([user_id])[0]
        already_rated_items = self.to_mask_items_mapping[mapped_user_id]

        all_item_ids = np.array(range(self.item_encoder.classes_.shape[0]))
        all_item_ids = np.setdiff1d(all_item_ids, already_rated_items)

        predictions = self._model.predict(
            x=[all_item_ids, np.full_like(all_item_ids, fill_value=mapped_user_id)]
        )
        top_predictions = sorted(
            zip(predictions, all_item_ids), key=lambda x: x[0], reverse=True
        )[: kwargs["k"]]
        top_predictions_ids = [pred[1] for pred in top_predictions]

        top_isbns = self.item_encoder.inverse_transform(top_predictions_ids)

        return list(top_isbns.astype(str))

    @classmethod
    def from_path(cls, model_dir):
        model = tf.keras.models.load_model(Path(model_dir, "explicit_base.model"))
        item_encoder_path = Path(model_dir, "explicit_book.encoder")
        user_encoder_path = Path(model_dir, "explicit_user.encoder")
        to_mask_items_mapping_path = Path(model_dir, "to_mask_items.mapping")

        item_encoder = joblib.load(item_encoder_path)
        user_encoder = joblib.load(user_encoder_path)
        to_mask_items_mapping = joblib.load(to_mask_items_mapping_path)
        return cls(model, item_encoder, user_encoder, to_mask_items_mapping)
