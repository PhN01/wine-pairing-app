import logging
import os

from app.utils.encryption import (
    decrypt_data,
    encrypt_data,
    get_crypt,
    load_and_decrypt_data,
    read_file,
    write_encrypted_data,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_get_crypt():
    crypt = get_crypt()
    assert hasattr(crypt, "encrypt")
    assert hasattr(crypt, "decrypt")


def test_read_file(example_wine_df):
    files = ["sample_file.csv", "sample_file.pkl", "sample_file.feather"]
    for file in files:
        ext = file.split(".")[-1]
        example_wine_df.__getattr__(
            {"csv": "to_csv", "pkl": "to_pickle", "feather": "to_feather"}[ext]
        )(file)
        try:
            df = read_file(file)
            df = df.reindex(columns=example_wine_df.columns)
            assert all(df.values == example_wine_df.values)
        except ValueError:
            assert True
        else:
            assert False
        os.remove(file)


def test_encrypt_decrypt(rand_arr_vector_list):
    for arr in rand_arr_vector_list:
        enc = encrypt_data(arr)
        dec = decrypt_data(enc)
        assert isinstance(dec, type(arr))
        assert dec.sum() == arr.sum()


def test_encrypt_write_and_load_data(example_wine_df):
    path = "test.pkl"
    enc = encrypt_data(example_wine_df)
    write_encrypted_data(enc, path)
    df = load_and_decrypt_data(path)
    os.remove("test.pkl")
    assert df.equals(example_wine_df)
