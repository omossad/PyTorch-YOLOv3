import numpy as np
pr = np.loadtxt('predicted.txt')
gt = np.loadtxt('labels.txt')
prox = 0
corr = 0
for i in range(len(pr)):
    for j in range(8):
        if pr[i][j] == gt[i][j]:
            if pr[i][j] == 1:
                corr = corr + 1
        else:
            if  j > 0 and j < 7:
                if pr[i][j] == gt[i][j-1]:
                    prox = prox + 1
                elif pr[i][j] == gt[i][j+1]:
                    prox = prox + 1
                else:
                    continue
            else:
                continue

print(len(pr))
print('proximity ' + str(prox))
print('ratio ' + str(prox/len(pr)))
print('correct ' + str(corr))
print('ratio ' + str(corr/len(pr)))
