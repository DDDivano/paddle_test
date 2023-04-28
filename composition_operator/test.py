a = "bn1.running_mean"
b = "bn1._mean"
c = "bn1.running_var"
d = "bn1._variance"
print(b.replace("_mean", "running_mean").replace("_variance", "running_var"))