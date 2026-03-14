import pandas as pd

def read_bonus_malus(val: str, m_value: int = -1):
    val_str = val.strip().upper()
    if val_str == 'M':
        return m_value
    try:
        return int(float(val_str))
    except:
        return pd.NA

def read_car_year(val: str, first_year=1900, last_year=2026):
    val_str = val.replace('\xa0', '').strip()
    try:
        val_int = int(float(val_str))
        if val_int < first_year or val_int > last_year:
            return pd.NA
        return val_int
    except:
        return pd.NA

def load_dataset(path, nrows = None):
    unused_columns = [
        "unique_id", 
        "is_individual_person", 
        "is_individual_person_name"
    ]
    
    df = pd.read_csv(
        path, 
        nrows=nrows,  
        usecols = lambda col: col not in unused_columns,
        dtype={
            'bonus_malus': 'str',
            'car_year': 'str',
            'ownerkato': 'Int64', 
            'ownerkato_short': 'Int64'
        },
        parse_dates=['operation_date'],
        date_format="%Y-%m-%d",
    )

    df.fillna({
        'claim_amount': 0, 
        'claim_cnt': 0
    }, inplace=True)
    
    return df

def remove_scores(df: pd.DataFrame) -> pd.DataFrame:
    score_cols_mask = df.columns.str.startswith('SCORE')
    return df.loc[:,~score_cols_mask]

def unique_columns(df, by):
    grp = df.groupby(by).agg("nunique")
    unique = []
    for column in grp.columns:
        if (grp[column] <= 1).all():
            unique.append(column)

    return unique

def non_unique_values(df, by, nu_column, *columns):
    grp = df.groupby(by).agg("nunique")
    non_unique = grp[grp[nu_column] > 1].index.to_frame(index=False)
    sortd = df.merge(non_unique, on=by, how='inner').sort_values(by=by)
    all_columns = by + [nu_column] + columns
    return sortd[all_columns]

def clean_outliers(df, column):
    # Calculate Q1, Q3, and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]