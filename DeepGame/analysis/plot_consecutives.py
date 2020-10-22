import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# matplotlib histogram
cons = np.loadtxt('consecutive_fixations.txt')
#outlier_threshold = 50
outlier_threshold = min(max(cons), 50)
cons = np.delete(cons*100,cons>outlier_threshold)
plt.hist(cons, color = 'blue', edgecolor = 'black',
         bins = int(outlier_threshold))

# seaborn histogram
sns.distplot(cons, hist=True, kde=False,
             bins=int(outlier_threshold), color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Similar Consecutive Fixations Count')
plt.xlabel('Time in milliseconds')
plt.ylabel('Fixations Count')
plt.savefig('hist.png')
