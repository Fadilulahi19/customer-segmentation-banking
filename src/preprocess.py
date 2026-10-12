import pandas as pd

def load_data(path):
    url = path
    response = urllib.request.urlopen(url)
    zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
    with zip_data.open('bank-full.csv') as f:
        df = pd.read_csv(f, sep=';')
    return df

def clean_data(df):
    df_clean = df.copy()

    # Handle 'unknown' categories 
    # Replace 'unknown' with mode for each column
    for col in ['job', 'education', 'contact', 'poutcome']:
    mode_val = df_clean[col][df_clean[col] != 'unknown'].mode()[0]
    df_clean[col] = df_clean[col].replace('unknown', mode_val)

    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)

    # Encode binary columns
    binary_map = {'yes': 1, 'no': 0}
    for col in ['default', 'housing', 'loan', 'y']:
    df_clean[col] = df_clean[col].map(binary_map)

    return df_clean