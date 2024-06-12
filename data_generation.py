import numpy as np
import pandas as pd

# For reproducibility
np.random.seed(42)

# Define number of samples
num_samples = 5

# Generate base features for normal operation
base_temp = np.random.normal(loc=50, scale=5, size=num_samples)
base_press = np.random.normal(loc=1500, scale=100, size=num_samples)
base_mmcfd = np.random.normal(loc=10, scale=2, size=num_samples)
base_bopd = np.random.normal(loc=500, scale=50, size=num_samples)
base_bwpd = np.random.normal(loc=200, scale=30, size=num_samples)
base_bsw = np.random.normal(loc=5, scale=1, size=num_samples)
base_co2 = np.random.normal(loc=1, scale=0.1, size=num_samples)
base_gas_grav = np.random.normal(loc=0.65, scale=0.05, size=num_samples)
base_cr = np.random.normal(loc=0.1, scale=0.02, size=num_samples)

# Initialize arrays for the features
temperature = np.zeros(num_samples)
pressure = np.zeros(num_samples)
mmcfd = np.zeros(num_samples)
bopd = np.zeros(num_samples)
bwpd = np.zeros(num_samples)
bsw = np.zeros(num_samples)
co2 = np.zeros(num_samples)
gas_grav = np.zeros(num_samples)
cr = np.zeros(num_samples)
labels = np.zeros(num_samples)

# Define leak severity levels
leak_levels = [0, 1, 2, 3]  # 0: No leak, 1: Minor leak, 2: Moderate leak, 3: Severe leak
proportions = [0.5, 0.2, 0.2, 0.1]  # Proportion of each severity level

# Generate data for each severity level
start_idx = 0
for level, proportion in zip(leak_levels, proportions):
    end_idx = start_idx + int(proportion * num_samples)
    if level == 0:
        # Normal operation
        temperature[start_idx:end_idx] = base_temp[start_idx:end_idx]
        pressure[start_idx:end_idx] = base_press[start_idx:end_idx]
        mmcfd[start_idx:end_idx] = base_mmcfd[start_idx:end_idx]
        bopd[start_idx:end_idx] = base_bopd[start_idx:end_idx]
        bwpd[start_idx:end_idx] = base_bwpd[start_idx:end_idx]
        bsw[start_idx:end_idx] = base_bsw[start_idx:end_idx]
        co2[start_idx:end_idx] = base_co2[start_idx:end_idx]
        gas_grav[start_idx:end_idx] = base_gas_grav[start_idx:end_idx]
        cr[start_idx:end_idx] = base_cr[start_idx:end_idx]
        labels[start_idx:end_idx] = 0
    else:
        # Leak conditions
        severity_factor = level * 0.2  # Increasing factor based on severity
        temperature[start_idx:end_idx] = base_temp[start_idx:end_idx] + severity_factor * np.random.normal(loc=5, scale=1, size=end_idx-start_idx)
        pressure[start_idx:end_idx] = base_press[start_idx:end_idx] - severity_factor * np.random.normal(loc=300, scale=50, size=end_idx-start_idx)
        mmcfd[start_idx:end_idx] = base_mmcfd[start_idx:end_idx] - severity_factor * np.random.normal(loc=3, scale=0.5, size=end_idx-start_idx)
        bopd[start_idx:end_idx] = base_bopd[start_idx:end_idx] - severity_factor * np.random.normal(loc=200, scale=30, size=end_idx-start_idx)
        bwpd[start_idx:end_idx] = base_bwpd[start_idx:end_idx] + severity_factor * np.random.normal(loc=100, scale=20, size=end_idx-start_idx)
        bsw[start_idx:end_idx] = base_bsw[start_idx:end_idx] + severity_factor * np.random.normal(loc=3, scale=0.5, size=end_idx-start_idx)
        co2[start_idx:end_idx] = base_co2[start_idx:end_idx] + severity_factor * np.random.normal(loc=0.5, scale=0.1, size=end_idx-start_idx)
        gas_grav[start_idx:end_idx] = base_gas_grav[start_idx:end_idx] + severity_factor * np.random.normal(loc=0.1, scale=0.02, size=end_idx-start_idx)
        cr[start_idx:end_idx] = base_cr[start_idx:end_idx] + severity_factor * np.random.normal(loc=0.2, scale=0.05, size=end_idx-start_idx)
        labels[start_idx:end_idx] = level
    start_idx = end_idx

# Create a DataFrame
data = pd.DataFrame({
    'Wellhead_Temp': temperature,
    'Wellhead_Press': pressure,
    'MMCFD_gas': mmcfd,
    'BOPD': bopd,
    'BWPD': bwpd,
    'BSW': bsw,
    'CO2_mol': co2,
    'Gas_Grav': gas_grav,
    'CR': cr,
    'leak_status': labels
})

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)


# Save the dataset to a CSV file
data.to_csv('pipeline_data.csv', index=False)
