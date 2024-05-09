import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('hyperparameter_tuning.txt', header=None, names=['var', 'uncertainty', 'raw_butterfly', 'actual', 'pred', 'actual2'], dtype=float)


df['amt_error'] = df['pred'] - df['actual']
df['abs_error'] = np.abs(df['pred'] - df['actual'])
# df['abs_error'] = df['amt_error'] # np.abs(df['pred'] - df['actual'])
df['percent_error'] = 100 * ((df['abs_error'] ) / df['actual'])
print(f'mean percent error {np.mean(df['percent_error'])}')

epsilon = 1e-99
# df['log_error'] = np.log((df['pred'] + epsilon) / (df['actual'] + epsilon))

df['uncertainty'] = np.minimum(df['uncertainty'], 100)



print(df)


# Calculate correlation coefficients
var_corr = np.log(df['var']).corr(df['abs_error'])
uncertainty_corr = np.log(df['uncertainty']).corr(df['abs_error'])

# # Print the correlation coefficients
print(f"correlation coefficient between log 'var' and error: {var_corr}")
print(f"correlation coefficient between log 'uncertainty' and error: {uncertainty_corr}")

# # Plotting var vs percent_error
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
# plt.scatter(df['var'], df['percent_error'], color='blue')
# plt.title('variance vs % error')
# plt.xlabel('variance')
# plt.ylabel('% error')

# # Plotting uncertainty vs percent_error
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
# plt.scatter(df['uncertainty'], df['percent_error'], color='blue')
# plt.title('uncertainty vs % error')
# plt.xlabel('uncertainty')
# plt.ylabel('% error')

# # Display the plots
# plt.tight_layout()
# plt.show()

print(df['amt_error'])
colors = ['green' if -5 < x < 5 else ('red' if x < 0 else 'blue') for x in df['amt_error']]

# Plotting var vs percent_error
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.scatter(np.log(df['var']), df['abs_error'], c=colors)
plt.title('log(variance) vs error')
plt.xlabel('log(variance)')
plt.ylabel('error')

# Plotting uncertainty vs percent_error
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.scatter(np.log(df['uncertainty']), df['abs_error'], c=colors)
plt.title('log(uncertainty) vs error')
plt.xlabel('log(uncertainty)')
plt.ylabel('error')

# Display the plots
plt.tight_layout()
plt.show()
