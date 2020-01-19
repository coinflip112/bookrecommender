from io import StringIO
from pathlib import Path

import isbnlib
import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

if __name__ == "__main__":
    # defining paths to raw files
    books_path = Path("data", "books.csv")
    users_path = Path("data", "users.csv")
    ratings_path = Path("data", "ratings.csv")

    # defining new header for raw imports
    books_headers = [
        "isbn",
        "title",
        "author",
        "publication_year",
        "publisher",
        "image_url_small",
        "image_url_medium",
        "image_url_large",
    ]
    users_headers = ["user_id", "location", "age"]
    ratings_headers = ["user_id", "isbn", "rating"]

    # types for columns
    books_dtypes = {
        "isbn": np.str_,
        "title": np.str_,
        "author": np.str_,
        "publication_year": np.int64,
        "image_url_small": np.str_,
        "image_url_medium": np.str_,
        "image_url_large": np.str_,
    }

    # types for users
    users_dtypes = {"user_id": np.int64, "age": np.float64}

    # reading books csv and applying transformations so that delimiters within string are correctly handled
    with open(file=books_path, mode="r", encoding="ISO-8859-1") as file_to_read:
        string_to_read = file_to_read.read()
        string_to_read = string_to_read.replace("&amp;", "&")
        string_to_read = string_to_read.replace('"; ', '" ')
        string_to_read = string_to_read.replace(" ; ", " ")
        string_to_read = string_to_read.replace("'", "")
        string_to_read = string_to_read.replace('Raag\\";\\"Free', 'Raag\\" \\"Free')
        string_to_read = string_to_read.replace('aders)\\"', "aders)")
        string_to_read = string_to_read.replace('Bergers\\"', "Bergers")
        ampersand_fixer = StringIO(string_to_read, newline=None)
        books = pd.read_csv(
            ampersand_fixer,
            sep=";",
            header=0,
            names=books_headers,
            encoding="ISO-8859-1",
            low_memory=False,
            quoting=0,
            dtype=books_dtypes,
        )

    # location converter
    location_converter = lambda x: x.split(",")[-1]

    # reading users csv
    users = pd.read_csv(
        filepath_or_buffer=users_path,
        sep=";",
        header=0,
        names=users_headers,
        encoding="ISO-8859-1",
        converters={"location": location_converter},
    )

    # renaming location to country
    users = users.rename({"location": "country"}, axis="columns")

    # reading ratings
    ratings = pd.read_csv(
        ratings_path, sep=";", header=0, names=ratings_headers, encoding="ISO-8859-1",
    )

    # converting to clean isbn13 format
    books = books.assign(isbn=books.isbn.parallel_apply(isbnlib.to_isbn13))
    books = books.dropna(subset=["isbn"])
    books = books.astype({"isbn": np.int64})

    # converting ratings to clean isbn13 format
    ratings = ratings.assign(isbn=ratings.isbn.parallel_apply(isbnlib.to_isbn13))
    ratings = ratings.dropna(subset=["isbn"])
    ratings = ratings.astype({"isbn": np.int64})

    # removing ratings for books which are not in db
    # assuming that only books in raw books import are recommendable
    unique_ratings_isbn = ratings.isbn.unique()
    unique_books_isbn = books.isbn.unique()
    rated_isbn_not_in_db = np.setdiff1d(
        unique_ratings_isbn, unique_books_isbn, assume_unique=True
    )
    ratings = ratings.loc[~ratings.isbn.isin(rated_isbn_not_in_db)]

    # defining paths to clean dataset
    books_clean_path = Path("data", "books_clean.csv")
    users_clean_path = Path("data", "users_clean.csv")
    ratings_clean_path = Path("data", "ratings_clean.csv")

    books.to_csv(books_clean_path, index=False)
    users.to_csv(users_clean_path, index=False)
    ratings.to_csv(ratings_clean_path, index=False)
