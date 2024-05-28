'''
This code gives 
- the GSR transition matrices
- the best guess after multiple shuffles
- the corresponding expected value
- graphs of the expected value over shuffles: the theoretical as well as the simulations 

'''
import matplotlib.pyplot as plt
import random
import math
from math import comb 
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
    sol = find_max_indices(A_m)
    return sol
def all_max_indices(n,m):
    A = np.array([[f(i, j,n) for j in range(1, n+1)] for i in range(1, n+1)]) 
    A_m = p(A,m)
    all_indices = []
    for col in range(A.shape[1]):
        column = A[:, col]
        max_value = np.max(column)
        max_indices = np.where(column == max_value)[0].tolist()
        all_indices.append(max_indices)
    return all_indices

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
#RIFFLE SHUFFLE (for simulations)
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
#
#
#GRAPHING SIMULATIONS
#
#
k = 10      #number of simulations
n =  52     #range of number of cards 
m = 7       #number of shuffles



    #Simulations + Theoretical values
x_values = range(1,m+1)
y_values = [mean_correct_guesses(m, best_guess(n,m), k) for m in x_values]
y_values2 = [ExpectedValue(n, m) for m in x_values]
plt.scatter(x_values, y_values, label=f"simulations")
plt.scatter(x_values, y_values2, label=f"theoretical")
    #Cutoff value: Bayer and Diaconis
#plt.axvline(x=(2/3*math.log2(n)), color='r', linestyle='-', label='Cutoff')


plt.xlabel('m, number of shuffles')
plt.ylabel('Expected Number of Correct Guesses')
plt.title(f"Expected Number of Correct Guesses Given {n} cards")
plt.grid(True)
plt.legend(loc = 'upper right')
plt.show()


