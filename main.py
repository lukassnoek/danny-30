import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

input = np.random.randn(1, 128)

model = load_model('DannyNet')
img = model.predict(input).squeeze()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig('mystery_image.png', bbox_inches='tight')
plt.close()
