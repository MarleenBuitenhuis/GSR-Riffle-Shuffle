"""
This code gives us
- GSR transition matrix
- Best guess strategy
- Corr. expected value 
- Graphs no. correct guesses as a function of cards (with parameter number of shuffles done)
    - for simulations
    - & theoretical distribution
"""
import matplotlib.pyplot as plt
import random
import math
import numpy as np
#
#
# TRANSITION MATRIX
#
#
def f(i,j,n):
    if i==j:
        return 2**(-i) + 2**(i-1-n)
    if i<j:
        return 2**(-j)*math.comb(j-1, j-i)
    else:
        return 2**(j-1-n)*math.comb(n-j, i-j)
def p(A,k):
    return np.linalg.matrix_power(A, k)
#
#
#The Best Guess 
#
#
def find_max_indices(matrix): #gives back the max column elements indices
    max_indices = np.argmax(matrix, axis=0)
    return [x for x in max_indices.tolist()] #Because in Thesis we use indices 1,2,...,n
def best_guess(n,m):
    A = np.array([[f(i, j,n) for j in range(1, n+1)] for i in range(1, n+1)]) 
    A_m = p(A,m)
    return find_max_indices(A_m)
#
#
#Expected Number Correct guesses Exact
#
#
def ExpectedValue(n,m):
    A = np.array([[f(i, j,n) for j in range(1, n+1)] for i in range(1, n+1)]) 
    A_m = p(A,m)
    l=  find_max_indices(A_m)
    sum = 0
    j=0
    for i in range(len(l)):
        sum+=A_m[l[i],j]
        j+=1        
    return sum
#
#
#RIFFLE SHUFFLE (for simmulations)
#
#
def riffle(l):
    n = len(l)
    outcomes = list(range(n + 1))
    probabilities = [(math.comb(n, k) / (2 ** n)) for k in outcomes]
    k = random.choices(outcomes, weights=probabilities,k=1)[0]
    if k ==0 or k == n:
        return(l)
    else:
        left_deck = l[:k+1]
        right_deck = l[k+1:]
    shuffled_deck = []
    while left_deck and right_deck:
        if random.random() < len(left_deck)/(len(right_deck)+len(left_deck)): 
            shuffled_deck.append(left_deck.pop(0))
        else:
            shuffled_deck.append(right_deck.pop(0))
    shuffled_deck.extend(left_deck)
    shuffled_deck.extend(right_deck) #this is to ensure the decks are used up fully
    return shuffled_deck
def riffle_shuffle(l,m):
    for i in range(m):
        l = riffle(l)
    return l
    #Checks the number of common elements of two lists
def correct_guesses(l,l2):
    count = 0
    for i in range(len(l)):
        if l[i] == l2[i]:
            count += 1
    return count
    #input: m shuffles, strategy list l2,k. Output: averge common elements over k "simulations"
def mean_correct_guesses(m,l2,k): 
    count = 0
    for i in range(k):
        l = list(range(len(l2)))
        l = riffle_shuffle(l, m)
        count+= correct_guesses(l,l2)
    return count /k
def E(x): #number of correct guesses according to theory
    return 2*math.sqrt(x/math.pi) 

#GRAPHING SIMULATIONS
k = 20 #number of simulations
n = 100 #range of number of cards 
m = 2  #number of shuffles
#Simulations
x_values = range(1,n+1)
y_values = [mean_correct_guesses(1,best_guess(n,1), k) for n in x_values]   #using matrix  and 1 shuffe
y_values1 = [mean_correct_guesses(m,best_guess(n,m), k) for n in x_values]  #using the matrix and m shuffles
plt.scatter(x_values, y_values, label=f'Simulations: Single Shuffle')    
plt.scatter(x_values, y_values1, label=f"Simulations: {m} Shuffles")
#Theoretical distri
xx = np.linspace(0, n, 10*n)  # Adjust the range and number of points as needed
xx2 = list(range(1,n))
yy = [E(x) for x in xx]
yy2= [ExpectedValue(n, m) for n in xx2]
plt.plot(xx, yy, color = 'r', label='Expectation: Single Shuffle')
plt.plot(xx2, yy2, color = 'b', label=f'Expectation: {m} Shuffles' )

plt.xlabel('n, number of cards')
plt.ylabel('correct guesses')
plt.title(f"Riffle Shuffle")
plt.grid(True)
plt.legend(loc = 'upper left')
plt.show()
