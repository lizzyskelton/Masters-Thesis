import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the additional data points, these are the measured values from EM31 data, an average of all data points in a 2m raduis
additional_x = [0.45, 0.76, 0.95, 1.20, 1.22, 1.17, -0.01, 0.00, 0.00, 0.95, 0.99, -0.23, 0.48, 0.48]  
additional_y = [1.92, 2.63, 2.79, 3.18, 3.21, 3.07, 0.93, 0.93, 0.92, 2.76, 2.76, 0.89, 2.61, 2.54] 

# Define the new set of data points, these are the expected values from the forward model 
new_x = [2.32, 3.06, 3.30, 4.29, 4.36, 4.00, 1.32, 1.35, 1.40, 3.87, 3.81, 1.29, 3.14, 3.44]  # Measured x (I) values from the EM31 data
new_y = [2.14, 2.85, 3.00, 3.52, 3.57, 3.32, 1.07, 1.11, 1.08, 3.42, 3.35, 1.06, 3.12, 3.16]  # Measured y (Q) values from the EM31 data 


# Plot 1: Inphase (additional_x vs new_x)
slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(additional_x, new_x)

plt.figure(figsize=(8, 6))
plt.scatter(additional_x, new_x, color='blue', label='Data Points', marker='o')
plt.plot(additional_x, [slope_x * x + intercept_x for x in additional_x], color='red', label=f'Fit: y={slope_x:.2f}x+{intercept_x:.2f}')
# plt.title("Inphase")
plt.xlabel("Inphase measured (x10^4 ppm)")
plt.ylabel("Inphase expected (x10^4 ppm)")
plt.legend()
plt.grid(True)
plt.show()  # Display the figure in a separate window

print(f"Inphase Analysis:")
print(f"  Slope: {slope_x:.2f}")
print(f"  Intercept: {intercept_x:.2f}")
print(f"  R-value: {r_value_x:.2f}")
print(f"  R-squared: {r_value_x**2:.2f}")
print(f"  P-value: {p_value_x:.2e}")
print(f"  Standard Error: {std_err_x:.2f}")

# Plot 2: Quadrature (additional_y vs new_y)
slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(additional_y, new_y)

plt.figure(figsize=(8, 6))
plt.scatter(additional_y, new_y, color='green', label='Data Points', marker='o')
plt.plot(additional_y, [slope_y * y + intercept_y for y in additional_y], color='red', label=f'Fit: y={slope_y:.2f}x+{intercept_y:.2f}')
# plt.title("Quadrature")
plt.xlabel("Quadrature measured (x10^4 ppm)")
plt.ylabel("Quadrature expected (x10^4 ppm)")
plt.legend() 
plt.grid(True)
plt.show()  # Display the figure in a separate window

print(f"Quadrature Analysis:")
print(f"  Slope: {slope_y:.2f}")
print(f"  Intercept: {intercept_y:.2f}")
print(f"  R-value: {r_value_y:.2f}")
print(f"  R-squared: {r_value_y**2:.2f}")
print(f"  P-value: {p_value_y:.2e}")
print(f"  Standard Error: {std_err_y:.2f}")


# Plot 3: Deviation from the expected value after various scaling factors 

# beginning with the linear regression 

# Define the 'scaled linear' deviation values - i.e. what percantage different the adapted measured values are from the expected values
scaled_linear_x = [0.14, 0.08, 0.13, 0.01, 0.02, 0.04, 0.21, 0.19, 0.15, 0.05, 0.01, 0.16, 0.15, 0.23]  
scaled_linear_y = [0.07, 0.07, 0.08, 0.02, 0.01, 0.05, 0.01, 0.01, 0.01, 0.09, 0.07, 0.02, 0.04, 0.09] 

slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(scaled_linear_x, scaled_linear_y)

plt.figure(figsize=(8, 6))
plt.scatter(scaled_linear_x, scaled_linear_y, color='blue', label='Data Points', marker='o')
# plt.plot(scaled_linear_x, [slope_x * x + intercept_x for x in scaled_linear_x], color='red', label=f'Fit: y={slope_x:.2f}x+{intercept_x:.2f}')
plt.title("Deviation from Expected Values for Measured Values Scaled with Linear Regression")
plt.xlabel("Inphase Deviation")
plt.ylabel("Quadrature Deviation")
# plt.legend()
plt.grid(True)
plt.show()  # Display the figure in a separate window

print(f"Linear Deviation Analysis:")
print(f"  Slope: {slope_x:.2f}")
print(f"  Intercept: {intercept_x:.2f}")
print(f"  R-value: {r_value_x:.2f}")
print(f"  R-squared: {r_value_x**2:.2f}")
print(f"  P-value: {p_value_x:.2e}")
print(f"  Standard Error: {std_err_x:.2f}")

print(scaled_linear_x)
print(scaled_linear_y)

avg_scaled_linear_x = sum(scaled_linear_x) / len(scaled_linear_x)
avg_scaled_linear_y = sum(scaled_linear_y) / len(scaled_linear_y)

print(f"Average scaled_linear_x: {avg_scaled_linear_x:.4f}")
print(f"Average scaled_linear_y: {avg_scaled_linear_y:.4f}")

