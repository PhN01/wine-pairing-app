import os
import pickle
from io import BytesIO
from typing import Any, Dict

import pandas as pd
from cryptography.fernet import Fernet

from app import constants as cons


def get_crypt() -> Fernet:
    """Get Fernet encryption engine with local key"""
    return Fernet(cons.FILE_KEY)


def read_file(filepath: str) -> pd.DataFrame:
    """Helper function for reading csv and pickle files

    Args:
        filepath (str): Path of file to be read

    Returns:
        pd.DataFrame: Data loaded from file
    """
    ext = filepath.split(".")[-1]
    if ext == "csv":
        return pd.read_csv(filepath)
    elif ext == "pkl":
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unknown file type {filepath}")


def encrypt_data(data: Any) -> str:
    """Pickle and encrypt a data object

    Args:
        data (Any): Data to be encrypted

    Returns:
        str: Encrypted file string
    """
    fernet = get_crypt()
    encoded = fernet.encrypt(pickle.dumps(data))
    return encoded


def decrypt_data(encoded: str) -> Any:
    """Decrypt a file string

    Args:
        encoded (str): Encrypted file string

    Returns:
        Any: Decrypted data
    """
    fernet = get_crypt()
    decoded = fernet.decrypt(encoded)
    dec_bytes = BytesIO(decoded)
    return pickle.load(dec_bytes)


def load_file_data() -> Dict[str, Any]:
    """Helper function to load the encrypted data used by this application

    Returns:
        Any: Dictionary mapping to decrypted data
    """
    with open(os.path.join(cons.DATA_PATH, cons.ENCRYPTED_DATA_FILE), "rb") as f:
        return decrypt_data(f.read())
