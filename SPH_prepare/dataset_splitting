import pandas as pd
from stratisfy import stratisfy_df

df = pd.read_csv('metadata.csv')
df['patient_id'] = df.Patient_ID
df = stratisfy_df(df, 'strat_fold')

print(df.columns)
print(df[:20])

# df.to_csv('new_sph_database.csv')





