import numpy as np  

value_a =  np.array([3, -0.5, 2, 7])
value_b =  np.array([2.5, 0.0, 2,2])


# MSE MAE formula is
mse = np.mean((value_a - value_b) ** 2)
mae = np.mean(np.abs(value_a - value_b))

print(mse,mae) # 6.375 1.5

# Create 1D arrays, indexing, slicing.
sample_array = np.array([7,2,5,0,12])
for index,value in enumerate(sample_array):
    # print(value)
    print(f"{index}->{sample_array[index]}")

part_a,part_b = sample_array[:3],sample_array[3:]
print(part_a,part_b)


# Do: mean, std, sum, element‑wise math (+, *, **2).
# Write mse_loss(y_true, y_pred) and mae_loss(y_true, y_pred) from scratch.
