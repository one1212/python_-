import numpy as np
import matplotlib.pyplot as plt
# 역슬래시(\) 대신 슬래시(/) 사용
file_path = 'C:/Users/Bit/Desktop/새 폴더 (2)/06/fruits_300.npy' 
fruits = np.load(file_path)
print(fruits.shape)
print(fruits[0, 0, :])
plt.imshow(fruits[0], cmap='gray')
#gray : 0에 가까울수록 어둡다.
#gray_r : 0에 가까울수록 밝다. 
plt.show()

fig, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape)
print(apple.mean(axis = 1))
plt.hist(apple.mean(axis = 1), alpha = 0.8, label='apple')
plt.hist(pineapple.mean(axis = 1), alpha = 0.8, label='pineapple')
plt.hist(banana.mean(axis = 1), alpha = 0.8, label='banana')
plt.legend()
plt.show()

fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].bar(range(10000), apple.mean(axis=0))
axs[1].bar(range(10000), pineapple.mean(axis=0))
axs[2].bar(range(10000), banana.mean(axis=0))
plt.show()

apple_mean = apple.mean(axis=0).reshape(100,100)
pineapple_mean = pineapple.mean(axis=0).reshape(100,100)
banana_mean = banana.mean(axis=0).reshape(100,100)
fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
apple_index = apple_index.reshape(10,10)
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i,j]], cmap='gray_r')
        axs[i,j].axis('off')
plt.show()