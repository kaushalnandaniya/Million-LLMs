# Model Checkpoints & Artifacts

This directory stores trained Learning-to-Rank (L2R) model bundles and precomputed embedding assets for fast inference.

## Generating the Production Router Bundle

To train the production XGBRanker with the optimal Bayesian parameters identified by Optuna and serialize it for live CLI routing:

```bash
python src/train_and_save_final.py
```

This will produce `models/xgb_enhanced_router.pkl`, which includes:
* Trained `xgb.XGBRanker` model fit on the full multi-modal feature space.
* Optimal residual blending parameters (`ce_blend`, `name_blend`).
* Complete feature column ordering.
