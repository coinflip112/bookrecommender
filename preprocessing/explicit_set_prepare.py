import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import joblib

if __name__ == "__main__":
    ratings = pd.read_csv("data/ratings_clean.csv")
    explicit_ratings = ratings.loc[ratings.rating != 0, :]
    explicit_ratings = shuffle(explicit_ratings)

    user_ids = explicit_ratings.user_id
    book_ids = explicit_ratings.isbn
    ratings = explicit_ratings.rating

    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()

    user_ids_encoded = user_encoder.fit_transform(user_ids)
    book_ids_encoded = book_encoder.fit_transform(book_ids)

    train_set_explicit = pd.DataFrame(
        data={
            "user_id": user_ids_encoded,
            "book_id": book_ids_encoded,
            "rating": ratings,
        }
    )
    train_set_explicit.to_csv("data/explicit_train_set.csv", index=False)

    joblib.dump(user_encoder, "encoders/explicit_user.encoder")
    joblib.dump(book_encoder, "encoders/explicit_book.encoder")
