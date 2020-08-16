import math

numColors = [7,7,7]
sumColors = sum(numColors)
P_Entropy = 0
for num in numColors:
    Pi = num/sumColors
    P_Entropy += Pi*math.log(Pi,2)

P_Entropy = -1*P_Entropy
print("P_Entropy" ,P_Entropy)

print("X2 = 3")
N = 5+1+6
print(N)
print("C_En top : 1.325")
print("C_En below : 1.224")
G = P_Entropy - ((N*1.325/sumColors) + (21-N)*1.244/sumColors)
print(G)

1.3221
1.0369