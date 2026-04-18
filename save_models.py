"""
save_models.py
Run this script AT THE END of your Milestone 1 notebook to save all trained
models and encoders so Milestone 2 can load them automatically.

Usage (in Colab/Jupyter at end of Milestone 1 code):
    exec(open('save_models.py').read())

Or copy-paste the lines into your notebook directly.
"""

import os
import pickle

os.makedirs('models', exist_ok=True)

models_to_save = {
    'models/rf_model.pkl':    rf_model,
    'models/lr_model.pkl':    lr_model,
    'models/scaler.pkl':      scaler,
    'models/le_city.pkl':     le_city,
    'models/le_property.pkl': le_property,
    'models/le_status.pkl':   le_status,
    'models/le_location.pkl': le_location,
}

for path, obj in models_to_save.items():
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'✅ Saved: {path}')

feature_order = X_train.columns.tolist()
with open('models/feature_order.pkl', 'wb') as f:
    pickle.dump(feature_order, f)
print(f'✅ Saved: models/feature_order.pkl')
print(f'   Features: {feature_order}')

print('\n🎉 All models saved! Copy the models/ folder to your real_estate_agent/ directory.')
print('   Then run: streamlit run app.py')