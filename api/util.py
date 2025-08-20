import pandas as pd

def file_to_df(upload):
    content = upload.file.read()
    return pd.read_csv(pd.io.common.BytesIO(content))
