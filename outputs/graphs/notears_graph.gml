graph [
  directed 1
  node [
    id 0
    label "scope_1_ghg_emissions"
  ]
  node [
    id 1
    label "scope_2_ghg_emissions"
  ]
  node [
    id 2
    label "scope_3_ghg_emissions"
  ]
  node [
    id 3
    label "emission_reduction_policy"
  ]
  node [
    id 4
    label "renewable_energy_share"
  ]
  node [
    id 5
    label "community_investment"
  ]
  node [
    id 6
    label "diversity_women_representation"
  ]
  node [
    id 7
    label "health_safety"
  ]
  node [
    id 8
    label "board_strategy_esg_oversight"
  ]
  node [
    id 9
    label "sustainable_finance_green_financing"
  ]
  node [
    id 10
    label "total_revenue"
  ]
  node [
    id 11
    label "reporting_quality"
  ]
  edge [
    source 0
    target 5
    weight 1.0
  ]
  edge [
    source 1
    target 0
    weight 1.0
  ]
  edge [
    source 1
    target 5
    weight 1.0
  ]
  edge [
    source 3
    target 8
    weight 1.0
  ]
  edge [
    source 4
    target 10
    weight 1.0
  ]
  edge [
    source 10
    target 9
    weight 1.0
  ]
  edge [
    source 11
    target 3
    weight 1.0
  ]
]
