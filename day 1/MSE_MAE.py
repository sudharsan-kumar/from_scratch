import numpy as np  

value_true =  np.array([3, -0.5, 2, 7])
value_pred =  np.array([2.5, 0.7, 2,2])


# MSE (Mean Squared Error) MAE (Mean Absolute Error) formula is
mse = np.mean((value_true - value_pred) ** 2)
mae = np.mean(np.abs(value_true - value_pred))

print(f"MSE: {mse}, MAE: {mae}") # 6.375 1.5

# Create 1D arrays, indexing, slicing.
sample_array = np.array([7,2,5,0,12])
for index,value in enumerate(sample_array):
    # print(value)
    print(f"In position {index} -> {sample_array[index]}")

part_a,part_b = sample_array[:3],sample_array[3:]
print(f"Slice at position 3: {part_a},{part_b}")

# print(value_true[0]+value_pred[0])
# print(value_true[0]-value_pred[0])
# print(value_true[0]*value_pred[0])
# print(value_true[0]/value_pred[0])

# print(np.add(value_true,value_pred))
# print(np.subtract(value_true,value_pred))
# print(np.divide(value_true,value_pred))
# print(np.multiply(value_true,value_pred))

# Do: mean, std, sum, element‑wise math (+, *, **2).
sample_mean =  (np.sum(sample_array))/len(sample_array)
print(f"Mean of sample_array: {sample_mean}")
# print(sample_array.mean())

temp_var = 0
for val in sample_array:
    temp_var += (val - sample_mean)**2
sample_std = (temp_var/(len(sample_array)-1))**(1/2)
print(f"Sample standard deviation of sample_array: {sample_std}")
# print(np.std(sample_array, ddof=1))

sample_sum = 0
for value in sample_array:
    sample_sum += value 

print(f"Sum of sample_array: {sample_sum}")

# print(np.sum(sample_array))

# Write mse_loss(y_true, y_pred) and mae_loss(y_true, y_pred) from scratch.

def mse_loss(y_true:np.array, y_pred:np.array):
    return np.mean((y_true-y_pred)**2)

# def mse_loss(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     diff = y_true - y_pred
#     return np.mean(diff * diff)

def mae_loss(y_true:np.array, y_pred:np.array):
    return np.mean(np.abs(y_true - y_pred))


print(f"MSE loss for value_true and value_pred: {mse_loss(value_true, value_pred)}")
print(f"MAE loss for value_true and value_pred: {mae_loss(value_true, value_pred)}")
