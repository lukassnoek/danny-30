import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.optimizers import Adam
from model import DannyNet, EarlyStoppingByLossVal


input_ = np.loadtxt('data/input.txt')
input_ = input_[None, :]  # add batch dim

output_ = load_img('data/output.jpg', color_mode="grayscale", target_size=(128, 128))
output_ = np.array(output_)[None, ...].astype(np.float32)  # add batch dim
output_ /= 255.

model = DannyNet(output_shape=output_.shape)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
model.fit(input_, output_, epochs=5000, callbacks=[EarlyStoppingByLossVal(value=0.001)])
model.save('DannyNet')