# Field Trial Plan — Randomized A/B Pilot

## Objective

Validate the AI crop recommendation system against current practices (extension officer advice or farmer intuition) in real-world conditions across 50–200 farmers.

## Study Design

- **Type**: Cluster-randomized controlled trial
- **Duration**: 2 full cropping seasons (1 Kharif + 1 Rabi = ~12 months)
- **Sample size**: 100–200 farmers across 4–6 districts in 2–3 Indian states
- **Allocation**: 50% treatment (AI recommendations), 50% control (standard practice)

## Participant Selection

### Inclusion Criteria
- Small-to-medium landholding farmers (0.5–5 hectares)
- Located in districts with diverse agro-climatic profiles
- Willing to provide soil samples and yield data
- Access to basic soil testing (government lab or portable kit)

### Exclusion Criteria
- Contract farming (fixed crop choice)
- Organic-only farms (different nutrient dynamics)
- Rainfed-only farms in drought-prone areas (unless specifically testing)

## Randomization

1. **Cluster selection**: Randomly select 8–12 village clusters
2. **Random allocation**: Assign clusters to treatment/control (cluster-level to avoid contamination)
3. **Stratification**: Balance by agro-climatic zone, farm size, and irrigation access
4. **Allocation ratio**: 1:1

## Protocol

### Treatment Group
1. Collect soil sample → send to lab
2. Enter lab results into app (field officer assisted)
3. Receive AI top-3 recommendation with explanations
4. Farmer makes final crop choice (recommendation is advisory only)
5. Record: chosen crop, planting date, inputs used

### Control Group
1. Collect soil sample → send to lab (same as treatment)
2. Farmer uses standard practice for crop choice
3. Record: chosen crop, planting date, inputs used

### Both Groups
- Record yield at harvest (kg per hectare, independently verified)
- Record: income from crop sale, total input costs
- Exit survey with farmer satisfaction

## Data Collection

| Data Point | Timing | Method |
|-----------|--------|--------|
| Soil sample results | Pre-planting | Lab report |
| Crop choice | Planting | Field officer visit |
| Inputs used (fertilizer, pesticide, water) | Mid-season | Field officer visit |
| Yield | Harvest | Measured + farmer report |
| Market price at sale | Post-harvest | Agmarknet + farmer report |
| Farmer satisfaction | Post-harvest | Structured survey |
| Weather conditions | Continuous | IMD station data |

## Outcome Metrics

### Primary
- **Yield difference**: Treatment vs control (kg/ha)
- **Net income difference**: Revenue minus input costs

### Secondary
- AI recommendation adherence rate
- Top-3 accuracy (was chosen crop in AI's top 3?)
- Farmer satisfaction score (1–5 scale)
- Crop diversity index

## Sample Size Justification

- Minimum detectable effect: 10% yield improvement
- Power: 80%, Significance: 5% (two-sided)
- ICC for cluster design: 0.05
- Required: ~80 farmers per arm → 160 total (200 with 20% attrition buffer)

## Ethics & Consent

- **IRB/Ethics approval**: Required before commencement
- **Informed consent**: Written in local language (see `docs/consent_template.md`)
- **Data privacy**: All farmer data pseudonymized; no PII stored
- **Voluntary participation**: Farmers may withdraw at any time
- **No harm**: AI provides advisory only; farmer retains full decision authority
- **Compensation**: Soil testing cost covered for all participants

## Timeline

| Phase | Duration | Activities |
|-------|----------|-----------|
| 1. Setup | Month 1–2 | Site selection, ethics approval, farmer recruitment |
| 2. Kharif season | Month 3–8 | Planting → harvest data collection |
| 3. Rabi season | Month 9–14 | Planting → harvest data collection |
| 4. Analysis | Month 15–16 | Statistical analysis, report writing |
| 5. Dissemination | Month 17 | Results publication, policy brief |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Farmer dropout | 20% over-enrollment, regular engagement |
| Crop failure (drought/flood) | Weather insurance, exclude force-majeure from analysis |
| Contamination (control sees treatment) | Cluster randomization |
| Data quality | Field officer training, spot checks, double data entry |
