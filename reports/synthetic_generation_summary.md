# Synthetic Data Generation Summary

## DAG Description

The synthetic dataset is generated from a fixed structural causal model
with known directed edges. Edge coefficients are sampled once per run
and reused across all requested sample sizes, so the data-generating
process is unchanged when only N changes.

- Variables: 28
- Edges: 53
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
| board_diversity                    | roa_eat                            | +             | 0.9696      |
| emission_reduction_policy_score    | roa_eat                            | +             | 0.6895      |
| emission_reduction_policy_score    | roe_eat                            | +             | 0.6299      |
| scope_1_emissions_tco2e            | debt_to_equity_ratio               | -             | -0.9757     |
| board_diversity                    | tobins_q                           | +             | 0.7269      |
| customer_satisfaction_score        | roa_eat                            | +             | 1.1698      |
| environmental_fines                | debt_to_equity_ratio               | +             | 0.9372      |
| iso_14001_exists                   | roa_eat                            | +             | 1.3327      |
| iso_14001_exists                   | tobins_q                           | +             | 1.2003      |
| scope_2_emissions_tco2e            | tobins_q                           | -             | -0.8124     |
| roa_eat                            | roe_eat                            | +             | 1.3323      |
| debt_to_equity_ratio               | roe_eat                            | +             | 1.3048      |
| tobins_q                           | pe_ratio                           | +             | 0.8875      |
| roa_eat                            | pe_ratio                           | +             | 0.7883      |
| roa_eat                            | asset_growth_pct                   | +             | 1.1825      |
| total_revenue_eur                  | asset_growth_pct                   | +             | 0.6398      |
| emission_reduction_policy_score    | pe_ratio                           | +             | 0.6999      |
| board_strategy_esg_oversight_score | asset_growth_pct                   | +             | 0.5074      |
| environmental_fines                | asset_growth_pct                   | -             | -1.2869     |

## Sample Statistics

### N = 110

- Data: `data/synthetic\synthetic_n110.csv`
- Metadata: `data/synthetic\synthetic_n110_metadata.json`
- Histogram figure: `reports/figures\synthetic_n110_histograms.png`

| variable                           | mean           | std            | min          | median        | max             |
| ---------------------------------- | -------------- | -------------- | ------------ | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.421          | 1.0          | 3.0           | 5.0             |
| renewable_energy_share             | 0.47           | 0.266          | 0.061        | 0.475         | 0.95            |
| scope_1_emissions_tco2e            | 483180.895     | 2180361.335    | 327.166      | 25564.803     | 17647280.933    |
| scope_2_emissions_tco2e            | 1963683.649    | 10563943.441   | 28.231       | 25189.241     | 106369363.91    |
| scope_3_emissions_tco2e            | 448760.481     | 930703.191     | 20775.807    | 203435.818    | 8856258.651     |
| total_energy_consumption           | 2320012.914    | 5175276.327    | 5122.869     | 643175.113    | 35127857.849    |
| environmental_fines                | 14.095         | 19.12          | 0.494        | 7.627         | 104.121         |
| iso_14001_exists                   | 0.536          | 0.501          | 0.0          | 1.0           | 1.0             |
| training_hours                     | 55.848         | 76.0           | 1.928        | 27.462        | 450.485         |
| injury_frequency_rate              | 2.896          | 1.77           | 0.0          | 2.841         | 6.912           |
| turnover_rate                      | 0.174          | 0.094          | 0.0          | 0.18          | 0.42            |
| diversity_representation           | 0.492          | 0.208          | 0.105        | 0.468         | 0.948           |
| community_investment_eur           | 9158148.948    | 26536850.449   | 14211.728    | 1442953.324   | 200541354.179   |
| customer_satisfaction_score        | 68.898         | 16.123         | 26.56        | 67.294        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.421          | 1.0          | 3.0           | 5.0             |
| board_diversity                    | 0.449          | 0.206          | 0.049        | 0.459         | 0.873           |
| ceo_chair_split                    | 0.5            | 0.502          | 0.0          | 0.5           | 1.0             |
| auditor_independence_score         | 49.006         | 28.307         | 0.0          | 48.694        | 100.0           |
| corruption_cases                   | 3.782          | 6.423          | 0.0          | 1.0           | 29.0            |
| total_asset                        | 1319452110.579 | 2999686589.639 | 10354000.112 | 369254750.531 | 22214074888.389 |
| total_revenue_eur                  | 1077833912.926 | 1703514379.862 | 3363981.0    | 412292829.14  | 8379869580.197  |
| roa_eat                            | 0.038          | 0.074          | -0.164       | 0.044         | 0.181           |
| debt_to_equity_ratio               | 2.597          | 4.469          | 0.368        | 0.855         | 20.086          |
| tobins_q                           | 3.371          | 3.31           | 0.135        | 1.51          | 7.389           |
| green_financing_eur                | 227292701.423  | 567127893.095  | 209811.603   | 34480035.692  | 4254811627.789  |
| pe_ratio                           | 23.442         | 23.026         | 1.0          | 11.059        | 54.598          |
| roe_eat                            | 0.011          | 0.144          | -0.3         | -0.002        | 0.4             |
| asset_growth_pct                   | 0.058          | 0.222          | -0.3         | 0.072         | 0.5             |

### N = 500

- Data: `data/synthetic\synthetic_n500.csv`
- Metadata: `data/synthetic\synthetic_n500_metadata.json`
- Histogram figure: `reports/figures\synthetic_n500_histograms.png`

| variable                           | mean           | std            | min         | median        | max             |
| ---------------------------------- | -------------- | -------------- | ----------- | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.416          | 1.0         | 3.0           | 5.0             |
| renewable_energy_share             | 0.495          | 0.246          | 0.026       | 0.503         | 0.986           |
| scope_1_emissions_tco2e            | 692388.456     | 3747803.593    | 53.853      | 35204.583     | 48271981.52     |
| scope_2_emissions_tco2e            | 3935938.049    | 33027613.396   | 4.203       | 35353.615     | 594672716.296   |
| scope_3_emissions_tco2e            | 514194.316     | 895829.218     | 11496.8     | 241420.692    | 8156724.801     |
| total_energy_consumption           | 3313644.681    | 16134263.707   | 12964.322   | 708159.557    | 319793611.656   |
| environmental_fines                | 17.448         | 30.634         | 0.24        | 7.941         | 365.701         |
| iso_14001_exists                   | 0.498          | 0.5            | 0.0         | 0.0           | 1.0             |
| training_hours                     | 60.472         | 87.45          | 0.862       | 29.542        | 723.676         |
| injury_frequency_rate              | 2.975          | 1.672          | 0.0         | 2.959         | 7.705           |
| turnover_rate                      | 0.179          | 0.098          | 0.0         | 0.179         | 0.476           |
| diversity_representation           | 0.504          | 0.21           | 0.044       | 0.51          | 0.969           |
| community_investment_eur           | 12381986.043   | 42805125.368   | 10929.757   | 2028652.42    | 743419777.089   |
| customer_satisfaction_score        | 69.853         | 15.575         | 12.434      | 70.298        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.416          | 1.0         | 3.0           | 5.0             |
| board_diversity                    | 0.4            | 0.199          | 0.028       | 0.385         | 0.921           |
| ceo_chair_split                    | 0.556          | 0.497          | 0.0         | 1.0           | 1.0             |
| auditor_independence_score         | 50.928         | 25.667         | 0.0         | 50.329        | 100.0           |
| corruption_cases                   | 2.94           | 5.485          | 0.0         | 0.0           | 25.0            |
| total_asset                        | 1638190617.819 | 3739660278.986 | 6911054.756 | 474203227.92  | 50884729497.432 |
| total_revenue_eur                  | 1735013404.456 | 4376161723.406 | 6533815.791 | 441528413.462 | 50985061915.563 |
| roa_eat                            | 0.032          | 0.075          | -0.215      | 0.034         | 0.24            |
| debt_to_equity_ratio               | 3.023          | 4.414          | 0.368       | 1.212         | 20.086          |
| tobins_q                           | 3.182          | 3.273          | 0.135       | 1.43          | 7.389           |
| green_financing_eur                | 183908176.454  | 399397418.89   | 374584.448  | 39107578.752  | 4273763903.203  |
| pe_ratio                           | 23.408         | 23.125         | 1.0         | 11.25         | 54.598          |
| roe_eat                            | 0.018          | 0.145          | -0.3        | 0.018         | 0.4             |
| asset_growth_pct                   | 0.053          | 0.214          | -0.3        | 0.054         | 0.5             |

### N = 2000

- Data: `data/synthetic\synthetic_n2000.csv`
- Metadata: `data/synthetic\synthetic_n2000_metadata.json`
- Histogram figure: `reports/figures\synthetic_n2000_histograms.png`

| variable                           | mean           | std            | min         | median        | max             |
| ---------------------------------- | -------------- | -------------- | ----------- | ------------- | --------------- |
| emission_reduction_policy_score    | 3.0            | 1.415          | 1.0         | 3.0           | 5.0             |
| renewable_energy_share             | 0.496          | 0.259          | 0.009       | 0.495         | 0.987           |
| scope_1_emissions_tco2e            | 632807.111     | 4694712.536    | 20.944      | 36903.859     | 158970385.967   |
| scope_2_emissions_tco2e            | 3047239.689    | 24748010.412   | 0.572       | 32490.766     | 791328166.353   |
| scope_3_emissions_tco2e            | 576822.419     | 990044.098     | 5850.161    | 270176.657    | 19146592.926    |
| total_energy_consumption           | 2413277.885    | 5871067.242    | 7962.959    | 771602.012    | 122874740.791   |
| environmental_fines                | 18.924         | 36.145         | 0.086       | 7.28          | 614.261         |
| iso_14001_exists                   | 0.502          | 0.5            | 0.0         | 1.0           | 1.0             |
| training_hours                     | 59.909         | 109.416        | 0.819       | 30.112        | 2334.396        |
| injury_frequency_rate              | 3.034          | 1.686          | 0.0         | 3.016         | 8.031           |
| turnover_rate                      | 0.182          | 0.099          | 0.0         | 0.181         | 0.514           |
| diversity_representation           | 0.505          | 0.204          | 0.049       | 0.506         | 0.952           |
| community_investment_eur           | 10933355.702   | 59821469.369   | 4508.805    | 1957691.253   | 2064178299.666  |
| customer_satisfaction_score        | 70.469         | 16.218         | 18.738      | 70.592        | 100.0           |
| board_strategy_esg_oversight_score | 3.0            | 1.415          | 1.0         | 3.0           | 5.0             |
| board_diversity                    | 0.424          | 0.203          | 0.019       | 0.413         | 0.949           |
| ceo_chair_split                    | 0.58           | 0.494          | 0.0         | 1.0           | 1.0             |
| auditor_independence_score         | 49.438         | 25.184         | 0.0         | 49.48         | 100.0           |
| corruption_cases                   | 3.062          | 5.704          | 0.0         | 0.0           | 31.0            |
| total_asset                        | 1509661673.405 | 3405726993.297 | 3750841.502 | 504703580.833 | 51553616935.979 |
| total_revenue_eur                  | 1690506776.501 | 4307169271.639 | 1188242.315 | 495102631.584 | 64354599045.11  |
| roa_eat                            | 0.033          | 0.078          | -0.25       | 0.036         | 0.246           |
| debt_to_equity_ratio               | 2.591          | 4.089          | 0.368       | 0.985         | 20.086          |
| tobins_q                           | 3.147          | 3.258          | 0.135       | 1.233         | 7.389           |
| green_financing_eur                | 218436948.164  | 587948116.67   | 131241.057  | 37468837.513  | 7956741639.89   |
| pe_ratio                           | 22.7           | 22.94          | 1.0         | 10.041        | 54.598          |
| roe_eat                            | 0.018          | 0.15           | -0.3        | 0.013         | 0.4             |
| asset_growth_pct                   | 0.048          | 0.216          | -0.3        | 0.053         | 0.5             |
