import matplotlib.pyplot as plt

relu_losses = [
0.2915, 0.1619, 0.1428, 0.2089, 0.0773,
0.0648, 0.1477, 0.0945, 0.0605, 0.0617,
0.0705, 0.0316, 0.0548, 0.0211, 0.0592,
0.0733, 0.0298, 0.0828, 0.0187, 0.0178
]

tanh_losses = [
0.2627, 0.3661, 0.2694, 0.4491, 0.3299,
0.3208, 0.1823, 0.1419, 0.0737, 0.1018,
0.1301, 0.0723, 0.1356, 0.0987, 0.1033,
0.0556, 0.1384, 0.1214, 0.0816, 0.0628
]

epochs = list(range(1, 21))

plt.figure(figsize=(8,5))

plt.plot(epochs, relu_losses, marker='o', label='ReLU')
plt.plot(epochs, tanh_losses, marker='s', label='Tanh')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("loss_curve.png", dpi=300)

print("已生成 loss_curve.png")