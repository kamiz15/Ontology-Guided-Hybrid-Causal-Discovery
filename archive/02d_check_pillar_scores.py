import pandas as pd
files = ['data/processed/data_ready.csv', 'data/synthetic/synthetic_n110.csv', 'data/synthetic/synthetic_n500.csv', 'data/synthetic/synthetic_n2000.csv']
pillar_cols = ['env_pillar_score', 'soc_pillar_score', 'gov_pillar_score', 'overall_esg_score']
for f in files:
    df = pd.read_csv(f)
    present = [c for c in pillar_cols if c in df.columns]
    missing = [c for c in pillar_cols if c not in df.columns]
    print(f'{f}: shape={df.shape}, present={len(present)}, missing={missing}')
