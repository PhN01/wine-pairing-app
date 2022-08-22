import os

import app.constants as cons
from app.utils.encryption import encrypt_data, read_file

if __name__ == "__main__":
    files = [
        os.path.join(cons.DATA_PATH, f)
        for f in os.listdir(cons.DATA_PATH)
        if os.path.isfile(os.path.join(cons.DATA_PATH, f))
        and not f == cons.ENCRYPTED_DATA_FILE
    ]

    data = {}

for f in files:
    fname = os.path.split(f)[-1]
    data[fname] = read_file(f)

    encoded = encrypt_data(data)

    with open(os.path.join(cons.DATA_PATH, cons.ENCRYPTED_DATA_FILE), "wb") as f:
        f.write(encoded)
