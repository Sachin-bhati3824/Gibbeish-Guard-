import numpy as np 

accepted_chars = "abcdefghijklmnopqrstuvwxyz "

def normalize_it(line):
    line = line.lower()
    noramlized = "" 
    for ch in line:
        if ch in accepted_chars:
            noramlized += ch 
    return noramlized



def n_gram(line,n):
    n_grams = [] 
    for i in range(len(line) - n+1):
        n_gram = line[i:(i+n)]
        n_grams.append(n_gram)
    return n_grams

k = int(len(accepted_chars)) 

counts = np.zeros((k,k))

char_to_idx= {
    ch: i 
    for i , ch in enumerate(accepted_chars)
}

with open("good.txt",'r') as file :
    for line in file :
        text = normalize_it(line)

        for i in range(len(text)-1):
            current_char = text[i]
            next_char = text[i+1] 
            row = char_to_idx[current_char] 
            col = char_to_idx[next_char] 

            counts[row][col] += 1 
probabilites = np.zeros((k,k))
for row in range(k):
    row_sum = counts[row].sum()
    probabilites[row] = (counts[row]+1) / (row_sum +k) 

print(probabilites.sum(axis=1))
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.imshow(probabilites)
plt.colorbar()

plt.xticks(range(len(accepted_chars)), accepted_chars)
plt.yticks(range(len(accepted_chars)), accepted_chars)

plt.xlabel("Next Character")
plt.ylabel("Current Character")
plt.title("Character Transition Counts")

plt.show()

log_probabilties = np.log(probabilites) 


def avg_transition_probability(text):
    text = normalize_it(text) 
    if(len(text)<2):
        return float("-inf")
    log_prob = 0 

    for i in range(len(text)-1):
        current_char = text[i] 
        next_char = text[i+1] 

        row = char_to_idx[current_char] 
        col = char_to_idx[next_char] 

        log_prob += log_probabilties[row][col] 

    return log_prob/ (len(text)-1) 



THRESHOLD = -3.3 
def is_gibberish(text):
    return avg_transition_probability(text) < THRESHOLD

print(is_gibberish("michel"))
print(is_gibberish("sflk sdff sfjd"))

plt.figure(figsize=(8,8))
plt.imshow(
    np.array(log_probabilties),
    cmap="viridis",
    aspect="auto"
)
plt.colorbar(label="log(count+1)")
plt.xticks(
    range(len(accepted_chars)),
    accepted_chars,
    rotation=90
)
plt.yticks(
    range(len(accepted_chars)),
    accepted_chars
)
plt.xlabel("Next Character")
plt.ylabel("Current Character")
plt.title("Bigram Transition Heatmap")

plt.tight_layout()
plt.show()