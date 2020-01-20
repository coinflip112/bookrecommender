from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


if __name__ == "__main__":
    ratings_path = Path("data", "ratings_clean.csv")
    ratings = pd.read_csv(ratings_path)
    explicit_ratings = ratings.loc[ratings.rating != 0]

    def get_and_save_mapping(values, filename):
        with open(filename, "w") as file_to_write:
            value_to_id = {
                value: value_id for value_id, value in enumerate(values.unique())
            }
            for value, value_id in value_to_id.items():
                file_to_write.write("{},{}\n".format(value, value_id))
        return value_to_id

    user_mapping = get_and_save_mapping(
        explicit_ratings["user_id"], "data/users_mapping.csv"
    )
    item_mapping = get_and_save_mapping(
        explicit_ratings["isbn"], "data/books_mapping.csv"
    )

    explicit_ratings = explicit_ratings.assign(
        visitor_id=explicit_ratings.loc[:, "user_id"].map(user_mapping.get),
        item_id=explicit_ratings.loc[:, "isbn"].map(item_mapping.get),
    )

    id_transformed_explicit_ratings = explicit_ratings[
        ["visitor_id", "item_id", "rating"]
    ]
    id_transformed_explicit_ratings.to_csv(
        path_or_buf="data/id_transformed_explicit_ratings.csv",
        index=False,
        header=False,
    )

    n_users, n_items = (
        id_transformed_explicit_ratings.visitor_id.nunique(),
        id_transformed_explicit_ratings.item_id.nunique(),
    )

    grouped_by_items = id_transformed_explicit_ratings.groupby("item_id")
    with tf.io.TFRecordWriter("data/users_for_item.tfrecords") as record_to_write:
        for item, grouped in grouped_by_items:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "key": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[item])
                        ),
                        "indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=grouped["visitor_id"].values
                            )
                        ),
                        "values": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=grouped["rating"].values
                            )
                        ),
                    }
                )
            )
            record_to_write.write(example.SerializeToString())

    grouped_by_users = id_transformed_explicit_ratings.groupby("visitor_id")
    with tf.io.TFRecordWriter("data/items_for_user.tfrecords") as record_to_write:
        for user, grouped in grouped_by_users:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "key": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[user])
                        ),
                        "indices": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=grouped["item_id"].values
                            )
                        ),
                        "values": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=grouped["rating"].values
                            )
                        ),
                    }
                )
            )
            record_to_write.write(example.SerializeToString())
