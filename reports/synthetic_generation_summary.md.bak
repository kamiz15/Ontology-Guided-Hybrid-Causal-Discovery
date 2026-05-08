# Synthetic Data Generation Summary

## DAG Description

The synthetic dataset is generated from a fixed structural causal model
with known directed edges. Edge coefficients are sampled once per run
and reused across all requested sample sizes, so the data-generating
process is unchanged when only N changes.

- Variables: 28
- Edges: 43
- Seed: 42
- Noise scale: 1.0

## Realism Clips

The SCM clips selected log-scale intermediates before exponentiation
to prevent unrealistic synthetic outliers while preserving the causal
signal carried by their parent variables.

- `tobins_q`: log value clipped to [-2.0, 2.0], giving a loose range of roughly [0.13, 7.39]
- `debt_to_equity_ratio`: log value clipped to [-1.0, 3.0], giving a loose range of roughly [0.37, 20.09]
- `corruption_cases`: Poisson log-rate clipped to [-3.0, 3.0], giving lambda in roughly [0.05, 20.09]

Real-world banking ranges informed these clip choices, but the clips
are deliberately loose to preserve recoverable synthetic signal.

## Variables

### Environmental

- `emission_reduction_policy_score` (ordinal)
- `renewable_energy_share` (continuous_0_1)
- `scope_1_emissions_tco2e` (log_normal)
- `scope_2_emissions_tco2e` (log_normal)
- `scope_3_emissions_tco2e` (log_normal)
- `total_energy_consumption` (log_normal)
- `environmental_fines` (log_normal)
- `iso_14001_exists` (binary)

### Social

- `training_hours` (log_normal)
- `injury_frequency_rate` (continuous)
- `turnover_rate` (continuous)
- `diversity_representation` (continuous_0_1)
- `community_investment_eur` (log_normal)
- `customer_satisfaction_score` (continuous)

### Governance

- `board_strategy_esg_oversight_score` (ordinal)
- `board_diversity` (continuous_0_1)
- `ceo_chair_split` (binary)
- `auditor_independence_score` (continuous_0_100)
- `corruption_cases` (count)

### Financial

- `total_asset` (log_normal_root)
- `total_revenue_eur` (log_normal)
- `roa_eat` (continuous)
- `debt_to_equity_ratio` (positive)
- `tobins_q` (positive)
- `green_financing_eur` (log_normal)
- `pe_ratio` (positive)
- `roe_eat` (continuous)
- `asset_growth_pct` (continuous)

## Ground-Truth Edges

| parent                             | child                              | expected_sign | coefficient |
| ---------------------------------- | ---------------------------------- | ------------- | ----------- |
| total_asset                        | total_revenue_eur                  | +             | 1.274       |
| total_asset                        | scope_1_emissions_tco2e            | +             | 0.9389      |
| total_asset                        | scope_2_emissions_tco2e            | +             | 1.3586      |
| total_asset                        | total_energy_consumption           | +             | 1.1974      |
| total_asset                        | training_hours                     | +             | 0.5942      |
| total_asset                        | community_investment_eur           | +             | 1.4756      |
| total_asset                        | debt_to_equity_ratio               | +             | 1.2611      |
| ceo_chair_split                    | board_strategy_esg_oversight_score | +             | 1.2861      |
| board_diversity                    | board_strategy_esg_oversight_score | +             | 0.6281      |
| board_strategy_esg_oversight_score | emission_reduction_policy_score    | +             | 0.9504      |
| board_strategy_esg_oversight_score | iso_14001_exists                   | +             | 0.8708      |
| board_strategy_esg_oversight_score | auditor_independence_score         | +             | 1.4268      |
| board_strategy_esg_oversight_score | green_financing_eur                | +             | 1.1439      |
| board_strategy_esg_oversight_score | corruption_cases                   | -             | -1.3228     |
| emission_reduction_policy_score    | renewable_energy_share             | +             | 0.9434      |
| emission_reduction_policy_score    | scope_1_emissions_tco2e            | -             | -0.7272     |
| emission_reduction_policy_score    | scope_2_emissions_tco2e            | -             | -1.0546     |
| emission_reduction_policy_score    | environmental_fines                | -             | -0.5638     |
| renewable_energy_share             | scope_2_emissions_tco2e            | -             | -1.3276     |
| iso_14001_exists                   | environmental_fines                | -             | -1.1317     |
| total_energy_consumption           | scope_1_emissions_tco2e            | +             | 1.2581      |
| total_energy_consumption           | scope_2_emissions_tco2e            | +             | 0.8545      |
| training_hours                     | injury_frequency_rate              | -             | -1.4707     |
| training_hours                     | turnover_rate                      | -             | -1.3931     |
| diversity_representation           | customer_satisfaction_score        | +             | 1.2784      |
| turnover_rate                      | customer_satisfaction_score        | -             | -0.6946     |
| auditor_independence_score         | corruption_cases                   | -             | -0.9667     |
| total_revenue_eur                  | roa_eat                            | +             | 0.5438      |
| roa_eat                            | tobins_q                           | +             | 0.6543      |
| debt_to_equity_ratio               | tobins_q                           | -             | -1.183      |
| emission_reduction_policy_score    | tobins_q                           | +             | 1.2448      |
| environmental_fines                | roa_eat                            | -             | -1.4675     |
| corruption_cases                   | tobins_q                           | -             | -0.8258     |
| renewable_energy_share             | green_financing_eur                | +             | 0.8705      |
| roa_eat                            | roe_eat                            | +             | 0.9696      |
| debt_to_equity_ratio               | roe_eat                            | +             | 0.6895      |
| tobins_q                           | pe_ratio                           | +             | 0.6299      |
| roa_eat                            | pe_ratio                           | +             | 0.9757      |
| roa_eat                            | asset_growth_pct                   | +             | 0.7269      |
| total_revenue_eur                  | asset_growth_pct                   | +             | 1.1698      |
| emission_reduction_policy_score    | pe_ratio                           | +             | 0.9372      |
| board_strategy_esg_oversight_score | asset_growth_pct                   | +             | 1.3327      |
| environmental_fines                | asset_growth_pct                   | -             | -1.2003     |

## Sample Statistics

### N = 110

- Data: `data/synthetic\synthetic_n110.csv`
- Metadata: `data/synthetic\synthetic_n110_metadata.json`
- Histogram figure: `reports/figures\synthetic_n110_histograms.png`

| variable                           | mean           | std            | min          | median        | max             |
| ---------------------------------- | -------------- | -------------- | ------------ | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.421          | 1.0          | 3.0           | 5.0             |
| renewable_energy_share             | 0.496          | 0.245          | 0.057        | 0.505         | 0.939           |
| scope_1_emissions_tco2e            | 883571.721     | 4547165.214    | 95.889       | 36453.287     | 37882448.387    |
| scope_2_emissions_tco2e            | 822759.475     | 2463981.29     | 4.281        | 16279.699     | 15297467.768    |
| scope_3_emissions_tco2e            | 446964.569     | 911292.358     | 20775.807    | 232525.761    | 8856258.651     |
| total_energy_consumption           | 2326737.622    | 5160858.968    | 5122.869     | 584036.797    | 35127857.849    |
| environmental_fines                | 18.348         | 29.727         | 0.205        | 8.469         | 207.114         |
| iso_14001_exists                   | 0.491          | 0.502          | 0.0          | 0.0           | 1.0             |
| training_hours                     | 55.884         | 72.933         | 1.928        | 27.95         | 450.485         |
| injury_frequency_rate              | 2.912          | 1.714          | 0.0          | 3.003         | 7.361           |
| turnover_rate                      | 0.171          | 0.096          | 0.0          | 0.175         | 0.427           |
| diversity_representation           | 0.484          | 0.216          | 0.105        | 0.46          | 0.948           |
| community_investment_eur           | 9085738.76     | 26509832.561   | 14211.728    | 1516646.462   | 200541354.179   |
| customer_satisfaction_score        | 68.824         | 16.948         | 20.545       | 67.741        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.421          | 1.0          | 3.0           | 5.0             |
| board_diversity                    | 0.439          | 0.196          | 0.049        | 0.448         | 0.845           |
| ceo_chair_split                    | 0.618          | 0.488          | 0.0          | 1.0           | 1.0             |
| auditor_independence_score         | 47.666         | 24.315         | 0.0          | 47.517        | 100.0           |
| corruption_cases                   | 3.473          | 6.192          | 0.0          | 0.0           | 27.0            |
| total_asset                        | 1396362676.173 | 3076165989.883 | 10354000.112 | 375657831.344 | 22214074888.389 |
| total_revenue_eur                  | 1142974904.329 | 1810987127.525 | 3363981.0    | 389574446.22  | 8379869580.197  |
| roa_eat                            | 0.043          | 0.05           | -0.093       | 0.047         | 0.163           |
| debt_to_equity_ratio               | 2.533          | 3.994          | 0.368        | 0.894         | 20.086          |
| tobins_q                           | 3.219          | 3.277          | 0.135        | 0.992         | 7.389           |
| green_financing_eur                | 247953246.563  | 525670808.61   | 804512.828   | 43471847.804  | 3725375375.597  |
| pe_ratio                           | 22.486         | 22.597         | 1.0          | 10.156        | 54.598          |
| roe_eat                            | 0.016          | 0.109          | -0.3         | 0.012         | 0.29            |
| asset_growth_pct                   | 0.046          | 0.238          | -0.3         | 0.053         | 0.5             |

### N = 500

- Data: `data/synthetic\synthetic_n500.csv`
- Metadata: `data/synthetic\synthetic_n500_metadata.json`
- Histogram figure: `reports/figures\synthetic_n500_histograms.png`

| variable                           | mean           | std            | min         | median        | max             |
| ---------------------------------- | -------------- | -------------- | ----------- | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.416          | 1.0         | 3.0           | 5.0             |
| renewable_energy_share             | 0.497          | 0.264          | 0.013       | 0.512         | 0.986           |
| scope_1_emissions_tco2e            | 690963.076     | 3779330.1      | 52.163      | 39498.561     | 63725349.384    |
| scope_2_emissions_tco2e            | 2520541.324    | 15817429.048   | 14.718      | 27125.539     | 264170874.477   |
| scope_3_emissions_tco2e            | 540903.295     | 943468.133     | 11496.8     | 245842.065    | 8156724.801     |
| total_energy_consumption           | 2679239.124    | 7789069.009    | 12964.322   | 697512.534    | 86263175.217    |
| environmental_fines                | 18.174         | 32.661         | 0.058       | 8.249         | 344.714         |
| iso_14001_exists                   | 0.492          | 0.5            | 0.0         | 0.0           | 1.0             |
| training_hours                     | 64.475         | 95.259         | 1.74        | 30.115        | 780.34          |
| injury_frequency_rate              | 3.018          | 1.747          | 0.0         | 3.03          | 8.181           |
| turnover_rate                      | 0.178          | 0.095          | 0.0         | 0.182         | 0.428           |
| diversity_representation           | 0.502          | 0.208          | 0.044       | 0.508         | 0.969           |
| community_investment_eur           | 12734835.322   | 43681825.454   | 10929.757   | 2051279.065   | 743419777.089   |
| customer_satisfaction_score        | 69.924         | 15.309         | 26.961      | 69.903        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.416          | 1.0         | 3.0           | 5.0             |
| board_diversity                    | 0.401          | 0.201          | 0.028       | 0.388         | 0.921           |
| ceo_chair_split                    | 0.554          | 0.498          | 0.0         | 1.0           | 1.0             |
| auditor_independence_score         | 49.245         | 25.644         | 0.0         | 49.606        | 100.0           |
| corruption_cases                   | 3.21           | 6.012          | 0.0         | 0.0           | 30.0            |
| total_asset                        | 1706121873.164 | 3865092463.217 | 6911054.756 | 470048950.482 | 50884729497.432 |
| total_revenue_eur                  | 1781859860.3   | 4451072824.935 | 1827216.256 | 454380660.184 | 50985061915.563 |
| roa_eat                            | 0.041          | 0.046          | -0.077      | 0.042         | 0.16            |
| debt_to_equity_ratio               | 2.802          | 4.394          | 0.368       | 0.966         | 20.086          |
| tobins_q                           | 3.102          | 3.138          | 0.135       | 1.495         | 7.389           |
| green_financing_eur                | 291443879.034  | 810907675.45   | 191495.133  | 40746157.552  | 9565602626.868  |
| pe_ratio                           | 23.569         | 23.434         | 1.0         | 9.768         | 54.598          |
| roe_eat                            | 0.014          | 0.109          | -0.282      | 0.01          | 0.361           |
| asset_growth_pct                   | 0.053          | 0.233          | -0.3        | 0.059         | 0.5             |

### N = 2000

- Data: `data/synthetic\synthetic_n2000.csv`
- Metadata: `data/synthetic\synthetic_n2000_metadata.json`
- Histogram figure: `reports/figures\synthetic_n2000_histograms.png`

| variable                           | mean           | std            | min         | median        | max             |
| ---------------------------------- | -------------- | -------------- | ----------- | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.415          | 1.0         | 3.0           | 5.0             |
| renewable_energy_share             | 0.499          | 0.264          | 0.008       | 0.504         | 0.98            |
| scope_1_emissions_tco2e            | 666650.218     | 4085675.655    | 20.25       | 35091.595     | 97218598.657    |
| scope_2_emissions_tco2e            | 2726001.031    | 20509986.386   | 1.275       | 26596.907     | 533457039.749   |
| scope_3_emissions_tco2e            | 582356.669     | 992814.099     | 5850.161    | 271235.593    | 19146592.926    |
| total_energy_consumption           | 2378395.573    | 5845219.047    | 7962.959    | 761459.568    | 122874740.791   |
| environmental_fines                | 17.454         | 30.801         | 0.058       | 7.171         | 375.371         |
| iso_14001_exists                   | 0.485          | 0.5            | 0.0         | 0.0           | 1.0             |
| training_hours                     | 60.043         | 110.103        | 0.819       | 30.029        | 2334.396        |
| injury_frequency_rate              | 3.042          | 1.687          | 0.0         | 3.031         | 8.041           |
| turnover_rate                      | 0.18           | 0.096          | 0.0         | 0.181         | 0.494           |
| diversity_representation           | 0.505          | 0.203          | 0.049       | 0.506         | 0.952           |
| community_investment_eur           | 10860447.278   | 59592438.097   | 4508.805    | 1925679.331   | 2064178299.666  |
| customer_satisfaction_score        | 69.964         | 15.775         | 13.091      | 70.501        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.415          | 1.0         | 3.0           | 5.0             |
| board_diversity                    | 0.424          | 0.204          | 0.019       | 0.413         | 0.949           |
| ceo_chair_split                    | 0.58           | 0.494          | 0.0         | 1.0           | 1.0             |
| auditor_independence_score         | 49.968         | 25.722         | 0.0         | 50.207        | 100.0           |
| corruption_cases                   | 3.298          | 6.192          | 0.0         | 0.0           | 37.0            |
| total_asset                        | 1475267365.391 | 3257465269.511 | 3750841.502 | 503668547.768 | 51553616935.979 |
| total_revenue_eur                  | 1679754684.978 | 4287617139.76  | 1188242.315 | 503144069.933 | 64354599045.11  |
| roa_eat                            | 0.041          | 0.046          | -0.157      | 0.043         | 0.169           |
| debt_to_equity_ratio               | 2.785          | 4.298          | 0.368       | 0.999         | 20.086          |
| tobins_q                           | 3.012          | 3.075          | 0.135       | 1.399         | 7.389           |
| green_financing_eur                | 231692249.302  | 593054838.773  | 142394.806  | 38782478.468  | 8704414976.779  |
| pe_ratio                           | 22.734         | 22.627         | 1.0         | 11.269        | 54.598          |
| roe_eat                            | 0.017          | 0.115          | -0.3        | 0.017         | 0.384           |
| asset_growth_pct                   | 0.049          | 0.229          | -0.3        | 0.045         | 0.5             |
