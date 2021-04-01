interval = 60
base = []
f = open('base.txt', 'r')
for l in f:
    base.append(int(l.split()[1]))
f.close()
#print(base)




deepGame = []
f = open('deepGame.txt', 'r')
for l in f:
    deepGame.append(int(l.split()[1]))
f.close()
#print(deepGame)
f_base = []
f_deepGame = []
count = 0
avg_base = 0
avg_deepGame = 0
for i in range(len(base)):
    avg_base += base[i]
    avg_deepGame += deepGame[i]
    if count % interval == 0 and count > 1:
        f_base.append(avg_base)
        f_deepGame.append(avg_deepGame)
        avg_base = 0
        avg_deepGame = 0
    count += 1

print(f_base)
#print(f_deepGame)
