import matplotlib.pyplot as plt

# epoch
epochs = list(range(1, 11))

# 你的实验数据
logistic_loss = [0.0891,0.0236,0.0171,0.0140,0.0122,0.0109,0.0100,0.0093,0.0087,0.0083]
perceptron_loss = [0.0125,0.0010,0.0007,0.0006,0.0005,0.0004,0.0004,0.0003,0.0003,0.0003]
svm_loss = [0.0377,0.0085,0.0063,0.0053,0.0047,0.0043,0.0041,0.0040,0.0037,0.0036]

# 画图
plt.figure()
plt.plot(epochs, logistic_loss, marker='o', label='Logistic Regression')
plt.plot(epochs, perceptron_loss, marker='o', label='Perceptron')
plt.plot(epochs, svm_loss, marker='o', label='SVM')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# 保存图片（可直接插入Word）
plt.savefig("loss_curve.png")

plt.show()