# Model Bundle

Place your trained model bundle here as:

```
nanocnc_model_bundle.pkl
```

Export it from your Kaggle notebook with:

```python
import joblib
joblib.dump(model_bundle, "nanocnc_model_bundle.pkl")
```

### Expected bundle structure

```python
{
  "length_model": sklearn.Pipeline,         # predicts CNC length (nm)
  "crystallinity_model": sklearn.Pipeline,  # predicts crystallinity (%)
  "feature_cols": [                        # exact column order used at training time
    "Acid_conc_wt_percent", "Temp_C", "Time_min",
    "Cellulose_Group_Wood / Pulp-based",
    "Cellulose_Group_Natural Plant Fiber",
    "Cellulose_Group_Agricultural Waste",
    "Cellulose_Group_Processed Cellulose",
    "Cellulose_Group_Algae / Marine",
    "Cellulose_Group_Other",
  ],
  "cellulose_groups": [                     # all group label strings
    "Wood / Pulp-based",
    "Natural Plant Fiber",
    "Agricultural Waste",
    "Processed Cellulose",
    "Algae / Marine",
    "Other",
  ],
  "model_name_length": str,        # displayed in the UI
  "model_name_crystallinity": str,
  "r2_length": float,
  "r2_crystallinity": float,
}
```

If this file is missing, the backend automatically runs in **Demo Mode**
using a linear approximation, so the UI still works.
