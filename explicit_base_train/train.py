import pandas as pd
import joblib
import tensorflow as tf
import argparse

if __name__ == "__main__":
    explicit_train_set = pd.read_csv("data/explicit_train_set.csv")
    book_encoder = joblib.load("encoders/explicit_book.encoder")
    user_encoder = joblib.load("encoders/explicit_user.encoder")
    n_items, n_users = book_encoder.classes_.shape[0], user_encoder.classes_.shape[0]

    def create_simple_cf_model(n_items, n_users, embedding_size):
        item_input = tf.keras.layers.Input(shape=(), name="item_input")
        user_input = tf.keras.layers.Input(shape=(), name="user_input")

        item_embedding = tf.keras.layers.Embedding(
            n_items + 1, embedding_size, name="item_embedding"
        )(item_input)
        user_embedding = tf.keras.layers.Embedding(
            n_users + 1, embedding_size, name="user_embedding"
        )(user_input)

        x = tf.keras.layers.multiply([item_embedding, user_embedding])
        x = tf.keras.layers.Dense(units=1)(x)

        model = tf.keras.Model(inputs=[item_input, user_input], outputs=x)
        model.compile(loss="mae", optimizer="adam", metrics=["mae"])
        return model

    model = create_simple_cf_model(n_items=n_items, n_users=n_users, embedding_size=2)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error", patience=20, restore_best_weights=True
    )
    model.fit(
        x=[explicit_train_set.book_id.values, explicit_train_set.user_id.values],
        y=explicit_train_set.rating.values,
        validation_split=0.1,
        callbacks=[early_stopping],
        epochs=100,
        batch_size=4196 * 16,
        verbose=1,
    )

    # Export the model to a SavedModel
    model.save(
        "explicit_model/explicit_base.model", save_format="h5",
    )
