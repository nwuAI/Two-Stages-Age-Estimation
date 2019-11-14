import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

number_x = np.arange(0,85,1)
y = stats.norm.pdf(number_x,52,2)
plt.plot(number_x,y,'r.')
plt.xlim(0,85)
plt.ylim(0,0.25)
plt.xlabel('Age')
plt.ylabel('Description Degree')
plt.text(30, .15, r'$\mu=52,\ \sigma=2$')