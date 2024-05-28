'''
This code contains
- Simulations of the riffle shuffle
- Graph: TVD of simulations to uniform distribution as a functino of time

'''

import random
import matplotlib.pyplot as plt
import math


'''Riffle Shuffle'''
def riffle(l):
    n = len(l)
    outcomes = list(range(n + 1))
    probabilities = [(math.comb(n, k) / (2 ** n)) for k in outcomes]
    k = random.choices(outcomes, weights=probabilities, k=1)[0]
    if k ==0 or k == n:
        return(l)
    else:
        left_deck = l[:k]
        right_deck = l[k:]
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

'''Calculates the Total Variation Distance'''
def count_elements(lst):
    element_counts = {}
    for element in lst:
        if tuple(element) in element_counts:
            element_counts[tuple(element)] += 1
        else:
            element_counts[tuple(element)] = 1
    return element_counts
def TVD(n,m,N):
    l=list(range(n))
    S_n = math.factorial(n)
    lst= []
    for i in range(N):
        lst.append(riffle_shuffle(l,m))
    d = count_elements(lst)
    distr = list(d.values())
    while len(distr) < S_n:
        distr.append(0)
    tvd = 0
    for i in range(S_n):
        tvd+= 0.5*abs(1/S_n - distr[i]/N)
    return tvd


'''Graphing'''
n = 7           #number of cards
N = 4900        #number of simulations (to get the distribution)
M = 15          #number of shuffles done

    #Coordinates of the Graph
x_values = range(M+1)  
y_values = [TVD(n, m, N) for m in x_values]  #Simulations 
plt.scatter(x_values, y_values, label='Simulations')

    #Cutoff Value
#plt.axvline(x=(2*math.log2(4*n/3)), color='r', linestyle='-', label='Mixing Time')

    #Expected Value Asymptotic
plt.axhline(y= (1-1/math.factorial(n))**N , color='red', linestyle='-', label='Expected Value')


    # Display the graph
plt.xlabel('t times shuffled')
plt.ylabel('tvd from uniform distribution')
plt.title(f"Shuffling of {n} cards using {N} simulations")
plt.grid(True)
plt.legend()
plt.show()


