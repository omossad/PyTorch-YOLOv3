import numpy as np
pr = np.loadtxt('predicted.txt')
gt = np.loadtxt('labels.txt')
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(pr)):
    for j in range(8):
        if pr[i][j] == gt[i][j]:
            if pr[i][j] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if pr[i][j] == 1:
                fp = fp + 1
            else:
                fn = fn + 1
precision = tp / (tp+fp)
recall = tp / (tp+fn)

print(len(pr))
print('True positive ' + str(tp))
print('False positive ' + str(fp))
print('True negative ' + str(tn))
print('False negative ' + str(fn))
print('Precision ' + str(precision))
print('Recall ' + str(recall))
