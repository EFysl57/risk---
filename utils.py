from typing import Optional
import pandas as pd

def read_bonus_malus(val: str, m_value: int = -1) -> Optional[int]:
    val_str = val.strip().upper()
    if val_str == 'M':
        return m_value
    try:
        return int(float(val_str))
    except:
        return None

def read_car_year(val: str, first_year=1900, last_year=2026) -> Optional[int]:
    val_str = val.replace('\xa0', '').strip()
    try:
        val_int = int(float(val_str))
        if val_int < first_year or val_int > last_year:
            return None
        return val_int
    except:
        return None

def load_dataset(path, nrows = None):
    df = pd.read_csv(path, nrows=nrows, 
                     converters={'bonus_malus': read_bonus_malus, 'car_year': read_car_year})
    
    # Pandas already have unique index
    assert df['unique_id'].nunique() == df.shape[0], f"{df['unique_id'].nunique()} != {df.shape[0]}"
    df.drop(columns="unique_id", inplace=True)

    # If no driver IIN just drop ???
    # We can also assume that this is new driver 
    df.dropna(subset=['driver_iin'], inplace=True)

    # Some numericals, that we can fill with zeros (probably)
    df.fillna({
        'claim_cnt': 0,
        'claim_amount': 0,
        'ownerkato_short': 0,
        'car_year': 0,
        'engine_volume': 0,
        'engine_power': 0,
    }, inplace=True)
    
    return df

def remove_scores(df: pd.DataFrame) -> pd.DataFrame:
    score_cols_mask = df.columns.str.startswith('SCORE')
    return df.loc[:,~score_cols_mask]

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Assume short KATO is enough (or drop short instead)
    df.drop(columns=['ownerkato'], inplace=True)

    # Drop name columns
    df.drop(columns=[
        'is_individual_person_name', 
        'is_residence_name', 
        'region_name', 
        'age_experience_name', 
        'vehicle_type_name'
    ], inplace=True)
    
    # Drop model of car (or drop mark instead)
    df.drop(columns=['model'], inplace=True)

    # New features
    # How many percent of premium we actually get
    df['term'] = (df['premium_wo_term'] / df['premium']) * 100

    return df

def check_dataset(df: pd.DataFrame, train=True):
    count_df = df.groupby(['contract_number', 'driver_iin', 'car_number']).size().reset_index().rename(columns={0:'count'})
    non_unique = count_df[count_df['count'] > 1]
    assert non_unique.size == 0, f"(policy, driver, car) is unique for each row"

    if train:
        result = df.groupby('contract_number')['claim_cnt'].nunique() <= 1
        assert result.all(), "Claim count is unique for policy"
    
        result = df.groupby('contract_number')['is_claim'].nunique() <= 1
        assert result.all(), "isClaim is unique for policy"

    result = df.groupby(['contract_number', 'driver_iin'])['bonus_malus'].nunique() <= 1
    assert result.all(), "Bonus malus is unique for (policy, driver)"

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