from dotenv import load_dotenv

load_dotenv()

import os
import s3fs


BUCKET_NAME = os.environ["BUCKET_NAME"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_HOST = os.environ["BUCKET_HOST"]

s3 = s3fs.S3FileSystem(
    endpoint_url=BUCKET_HOST, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY
)

with s3.open(f"{BUCKET_NAME}/new-file", "wb") as f:
    f.write(2 * 2**20 * b"a")
    f.write(2 * 2**20 * b"a")
    f.write(2 * 2**20 * b"a")  # data is flushed and file closed
    
print(s3.du(f"{BUCKET_NAME}/new-file"))
