accepted_chars = "abcdefghijklmnopqrstuvwxyz0123456789-.,()'\" "

import pickle 



def normalize_it(line):
    line = line.lower()
    noramlized = "" 
    for ch in line:
        if ch in accepted_chars:
            noramlized += ch 
    return noramlized


word_counts = {} 
with open("eng_sentences.tsv",'r',encoding="utf-8") as file:
    for line in file: 
        line = line.strip().split("\t")
        sentence = line[2]
        words = sentence.lower().split()
        for word in words : 
            word = normalize_it(word) 
            if word =="" :
                continue
            word_counts[word] = word_counts.get(word,0) + 1 

english_words = set()

for word,count in word_counts.items():
    if count>=5:
        english_words.add(word)


with open("vocabulary.pkl","wb")as f:
    pickle.dump(english_words,f)


         
