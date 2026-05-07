import pandas as pd
df = pd.read_csv('data/synthetic/synthetic_n2000.csv')
pillar_cols = [c for c in ['env_pillar_score', 'soc_pillar_score', 'gov_pillar_score', 'overall_esg_score'] if c in df.columns]
print(df[pillar_cols].describe())
