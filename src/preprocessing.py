from collections import deque
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def strict_past(data, keys, columns):
    positions = pd.Series(np.arange(len(data)), index=data.index)
    first = positions.groupby([data[key] for key in keys], observed=True).transform('min')
    data[columns] = data[columns].iloc[first.to_numpy()].set_axis(data.index)

def previous_expanding(data, column, statistic):
    grouped = data.groupby('cc_num', observed=True)[column]
    values = getattr(grouped.expanding(), statistic)()
    values = values.groupby(level=0, observed=True).shift(1)
    
    return values.reset_index(level=0, drop=True).sort_index()


def unique_values_in_previous_days(data, column, days):
    result = np.zeros(len(data), dtype='int32')

    for _, group in data.groupby('cc_num', sort=False, observed=True):
        events = deque()
        frequencies = {}
        positions = group.index.to_numpy()
        times = group['trans_date_trans_time'].to_numpy()
        values = group[column].to_numpy()
        start = 0

        while start < len(group):
            end = start + 1

            while end < len(group) and times[end] == times[start]:
                end += 1

            cutoff = times[start] - np.timedelta64(days, 'D')
            
            while events and events[0][0] < cutoff:
                _, value = events.popleft()
                frequencies[value] -= 1

                if frequencies[value] == 0:
                    del frequencies[value]
                    
            result[positions[start:end]] = len(frequencies)

            for i in range(start, end):
                if pd.notna(values[i]):
                    events.append((times[i], values[i]))
                    frequencies[values[i]] = frequencies.get(values[i], 0) + 1
                    
            start = end

    return result


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_transactions(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset does not exist: {path}')
    df = pd.read_csv(path, dtype={'cc_num': 'string', 'merchant': 'string', 'category': 'string'})
    loaded = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    return df, loaded, duplicates


def validate_transactions(df):
    required = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
        'lat', 'long', 'merch_lat', 'merch_long', 'gender', 'dob', 'job',
        'city_pop', 'state', 'is_fraud',
    ]

    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')

    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')

    if df['trans_date_trans_time'].isna().any() or df['cc_num'].isna().any():
        raise ValueError('Every transaction needs a valid timestamp and card number for historical features.')

    if (~np.isfinite(df['amt']) | (df['amt'] < 0)).any():
        raise ValueError('Investigate missing, non-finite, or negative amounts before calculating features.')

    if not df['is_fraud'].isin([0, 1]).all():
        raise ValueError('Investigate missing or invalid fraud labels before analysis.')

    return df


def sort_transactions(df):
    df['row_id'] = np.arange(len(df))

    df = df.sort_values(['cc_num', 'trans_date_trans_time', 'row_id']).reset_index(drop=True)

    return df


def add_temporal_features(df):
    ts = df['trans_date_trans_time']

    df['hour'] = ts.dt.hour

    df['day_of_week'] = ts.dt.dayofweek

    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    df['is_night'] = df['hour'].between(0, 5).astype('int8')

    df['is_evening'] = df['hour'].between(18, 23).astype('int8')

    df['month'] = ts.dt.month

    df['day_of_month'] = ts.dt.day

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)

    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_amount_features(df):
    df['log_amt'] = np.log1p(df['amt'])

    df['customer_history_count'] = df.groupby('cc_num', observed=True).cumcount()

    df['customer_avg_amt'] = previous_expanding(df, 'amt', 'mean')

    df['customer_median_amt'] = previous_expanding(df, 'amt', 'median')

    df['customer_max_amt'] = previous_expanding(df, 'amt', 'max')

    df['customer_std_amt'] = previous_expanding(df, 'amt', 'std')

    amount_history = ['customer_history_count', 'customer_avg_amt', 'customer_median_amt', 'customer_max_amt', 'customer_std_amt']

    strict_past(df, ['cc_num', 'trans_date_trans_time'], amount_history)

    df['has_customer_history'] = (df['customer_history_count'] > 0).astype('int8')

    df['amount_vs_avg'] = df['amt'] / df['customer_avg_amt'].replace(0, np.nan)

    df['amount_diff_from_avg'] = df['amt'] - df['customer_avg_amt']

    df['amount_vs_median'] = df['amt'] / df['customer_median_amt'].replace(0, np.nan)

    df['amount_zscore'] = (df['amt'] - df['customer_avg_amt']) / df['customer_std_amt'].replace(0, np.nan)

    return df


def add_behavior_features(df):
    df['_transaction'] = 1

    grouped = df.set_index('trans_date_trans_time').groupby('cc_num', observed=True)

    rolling_columns = []

    for label, window in [('1h', '1h'), ('24h', '24h'), ('7d', '7D'), ('30d', '30D')]:
        name = f'tx_count_{label}'
        df[name] = grouped['_transaction'].rolling(window, closed='left').sum().reset_index(drop=True).fillna(0).astype('int32')
        rolling_columns.append(name)

    for label, window in [('1h', '1h'), ('24h', '24h'), ('7d', '7D'), ('30d', '30D')]:
        name = f'tx_amount_sum_{label}'
        df[name] = grouped['amt'].rolling(window, closed='left').sum().reset_index(drop=True).fillna(0)
        rolling_columns.append(name)

    df['time_since_last_tx'] = df.groupby('cc_num', observed=True)['trans_date_trans_time'].diff().dt.total_seconds() / 60

    rolling_columns.append('time_since_last_tx')

    strict_past(df, ['cc_num', 'trans_date_trans_time'], rolling_columns)

    df['tx_amount_avg_30d'] = df['tx_amount_sum_30d'] / df['tx_count_30d'].replace(0, np.nan)

    df['unique_merchants_7d'] = unique_values_in_previous_days(df, 'merchant', 7)

    df['unique_categories_7d'] = unique_values_in_previous_days(df, 'category', 7)

    df['unique_categories_30d'] = unique_values_in_previous_days(df, 'category', 30)

    for source, feature in [('merchant', 'is_new_merchant'), ('category', 'is_new_category')]:
        first_seen = df.groupby(['cc_num', source], observed=True)['trans_date_trans_time'].transform('min')
        df[feature] = (df['trans_date_trans_time'] == first_seen).astype('Int8').where(df[source].notna())

    df = df.drop(columns='_transaction')

    return df


def add_geographic_features(df):
    valid_coordinates = (
        df['lat'].between(-90, 90) & df['long'].between(-180, 180)
        & df['merch_lat'].between(-90, 90) & df['merch_long'].between(-180, 180)
    )

    df['customer_merchant_distance_km'] = haversine_km(
        df['lat'].to_numpy(), df['long'].to_numpy(), df['merch_lat'].to_numpy(), df['merch_long'].to_numpy()
    )

    df.loc[~valid_coordinates, 'customer_merchant_distance_km'] = np.nan

    df['avg_customer_distance'] = previous_expanding(df, 'customer_merchant_distance_km', 'mean')

    distance_grouped = df.set_index('trans_date_trans_time').groupby('cc_num', observed=True)

    df['max_distance_7d'] = distance_grouped['customer_merchant_distance_km'].rolling(
        '7D', closed='left'
    ).max().reset_index(drop=True)

    strict_past(df, ['cc_num', 'trans_date_trans_time'], ['avg_customer_distance', 'max_distance_7d'])

    df['distance_vs_avg'] = df['customer_merchant_distance_km'] / df['avg_customer_distance'].replace(0, np.nan)

    return df, valid_coordinates


def add_merchant_features(df):
    merchant_history = df.loc[df['merchant'].notna(), ['row_id', 'merchant', 'trans_date_trans_time']].copy()

    merchant_history = merchant_history.sort_values(['merchant', 'trans_date_trans_time', 'row_id']).reset_index(drop=True)

    merchant_history['merchant_tx_count_history'] = merchant_history.groupby('merchant', observed=True).cumcount()

    strict_past(merchant_history, ['merchant', 'trans_date_trans_time'], ['merchant_tx_count_history'])

    df = df.merge(merchant_history[['row_id', 'merchant_tx_count_history']], on='row_id', how='left', validate='one_to_one')

    return df


def add_demographic_features(df):
    dob = pd.to_datetime(df['dob'], errors='coerce')

    df['age'] = (df['trans_date_trans_time'] - dob).dt.days / 365.25

    df.loc[~df['age'].between(0, 120), 'age'] = np.nan

    return df


def create_features(df):
    df = sort_transactions(df)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_behavior_features(df)
    df, _ = add_geographic_features(df)
    df = add_merchant_features(df)

    return add_demographic_features(df)



def quality_filter(X_train, MISSING_REVIEW=0.60, MISSING_LIMIT=0.80, VARIANCE_LIMIT=0.01, CORRELATION_LIMIT=0.95):
    missing_rate = X_train.isna().mean()

    quality_report = pd.DataFrame({'feature': X_train.columns, 'missing_pct': missing_rate.to_numpy() * 100})

    quality_report['review_missing'] = quality_report['missing_pct'] > MISSING_REVIEW * 100

    removed = []

    kept = []

    fingerprints = {}

    for column in X_train.columns:
        values = X_train[column]

        if missing_rate[column] > MISSING_LIMIT:
            removed.append((column, 'missing > 80%'))
            continue

        if pd.api.types.is_numeric_dtype(values):
            comparable = values.astype('float64')

            if np.isinf(comparable).any():
                raise ValueError(f'Feature {column} contains infinite values; investigate before selection.')
        
            if comparable.var(ddof=0) <= VARIANCE_LIMIT:
                removed.append((column, 'variance <= 0.01'))
                continue
        else:
            comparable = values.astype('string')
        
            if comparable.nunique(dropna=False) <= 1:
                removed.append((column, 'constant category'))
                continue
            
        fingerprint = sha256(pd.util.hash_pandas_object(comparable, index=False).to_numpy().tobytes()).digest()
        duplicate = None
    
        for prior in fingerprints.get(fingerprint, []):
            prior_values = X_train[prior].astype(comparable.dtype)
        
            if comparable.equals(prior_values):
                duplicate = prior
                break
    
        if duplicate is not None:
            removed.append((column, f'duplicate of {duplicate}'))
            continue
    
        fingerprints.setdefault(fingerprint, []).append(column)
        kept.append(column)

    numeric_kept = [column for column in kept if pd.api.types.is_numeric_dtype(X_train[column])]

    train_correlation = X_train[numeric_kept].corr().abs()

    correlation_dropped = set()

    for i, column in enumerate(numeric_kept):
        if column in correlation_dropped:
            continue
    
        for other in numeric_kept[i + 1:]:
            if other not in correlation_dropped and train_correlation.loc[column, other] > CORRELATION_LIMIT:
                correlation_dropped.add(other)
                removed.append((other, f'correlation > 0.95 with {column}'))

    filtered_features = [column for column in kept if column not in correlation_dropped]

    return quality_report, removed, filtered_features


def mutual_information_selection(X_train, y_train, filtered_features, N_FEATURES=5):
    if len(filtered_features) < N_FEATURES:
        raise ValueError(f'Only {len(filtered_features)} features remain; cannot select {N_FEATURES}. Review the filters.')

    scores = []

    cyclic_features = {'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'}

    for column in filtered_features:
        values = X_train[column]

        if column in cyclic_features:
            discrete = True
            encoded, _ = pd.factorize(values.round(12), sort=True, use_na_sentinel=False)
        elif pd.api.types.is_numeric_dtype(values):
            discrete = pd.api.types.is_integer_dtype(values) or pd.api.types.is_bool_dtype(values)
            numeric = values.astype('float64')
            encoded = numeric.fillna(numeric.median()).to_numpy()
        else:
            discrete = True
            encoded, _ = pd.factorize(values, sort=True, use_na_sentinel=False)

        score = mutual_info_classif(encoded.reshape(-1, 1), y_train.to_numpy(),
                                     discrete_features=[discrete], random_state=42)[0]
        scores.append((column, score))

    mi_scores = pd.DataFrame(scores, columns=['feature', 'mutual_information'])

    mi_scores = mi_scores.sort_values('mutual_information', ascending=False, kind='stable').reset_index(drop=True)

    selected_features = mi_scores.head(N_FEATURES)['feature'].tolist()

    return mi_scores, selected_features
