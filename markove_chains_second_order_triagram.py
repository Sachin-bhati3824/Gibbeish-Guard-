import numpy as np 
accepted_chars = "abcdefghijklmnopqrstuvwxyz0123456789-.,()'\" "
import sys 

def normalize_it(line):
    line = line.lower()
    noramlized = "" 
    for ch in line:
        if ch in accepted_chars:
            noramlized += ch 
    return noramlized

def n_gram(text,n):
    n_grams = [] 
    for i in range(len(text)- n+1):
        n_gram = text[i:(i+n)]
        n_grams.append(n_gram)

    return n_grams 

k = int(len(accepted_chars))

counts = np.zeros((k,k,k))

char_to_idx= {
    ch: i 
    for i , ch in enumerate(accepted_chars)
}
proccessed_char = 0 
with open("eng_sentences.tsv",'r',encoding="utf-8") as file :
    for line in file :

        cols = line.strip().split("\t")
        sentence = cols[2]
        progress = (int(cols[0])/12943143 ) * 100 
        print(f"\rPROGRESS: {progress:.6f}%", end ="")
        sys.stdout.flush()
        text = normalize_it(sentence)
        
        for i in range(1,len(text)-1):
            current_char = text[i]
            next_char = text[i+1] 
            last_char = text[i-1] 
            row = char_to_idx[current_char] 
            col = char_to_idx[next_char] 
            z = char_to_idx[last_char]

            counts[z][row][col]  += 1 

np.save("trigram_counts.npy",counts)
# import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# z, y, x = np.indices(counts.shape)

# mask = counts > 0

# ax.scatter(
#     x[mask],
#     y[mask],
#     z[mask],
#     c=np.log1p(counts[mask]),
#     cmap="viridis",
#     s=np.log1p(counts[mask]) * 10
# )
# plt.show() 

probabilties = np.zeros_like(counts)

for current in range(k):
    for prev in range(k): 
        total = counts[prev,current,:].sum()
        probabilties[prev,current,:] = (
            counts[prev,current,:] + 1
        ) / ( 
            total + k 
        )
    

log_probabilities = np.log(probabilties)
np.save("trigram_log_probs.npy",log_probabilities)


def avg_transition_probability(text):

    text = normalize_it(text)

    if len(text) < 3:
        return float("-inf")

    score = 0

    for i in range(1, len(text)-1):

        prev_char = text[i-1]
        curr_char = text[i]
        next_char = text[i+1]

        prev = char_to_idx[prev_char]
        curr = char_to_idx[curr_char]
        nxt = char_to_idx[next_char]

        score += log_probabilities[prev][curr][nxt]

    return score / (len(text)-2)

THRESHOLD = -3.1



# with open("test.txt","r") as file :
#     for line in file : 
#         lista = line.strip()

# tests = [
#     "minecraft",
#     "i like minecraft",
#     "i really like minecraft"
# ]

# for t in tests:
#     print(t, avg_transition_probability(t))