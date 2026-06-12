gibberish = [] 
correct = [] 

with open("gibbberish_words.txt","r",encoding="utf-8") as file :
    for line in file :
        gibberish.append(line.strip())
        
with open("correct_words.txt","r",encoding="utf-8") as file :
    for line in file :
        correct.append(line.strip())

import random 
with open("hybrid_gibberish.txt","w",encoding="utf-8") as file :
    for i in range(6500):
        lista = [] 
        amount_of_good = random.randint(1,10)
        for j in range(amount_of_good):
            word = random.choice(correct)
            lista.append(word)
        
        amount_of_gibberish = random.randint(1,4)
        for i in range(amount_of_gibberish):
            word = random.choice(gibberish)
            lista.append(word)
        random.shuffle(lista)
        sentence = " ".join(lista)
        sentence += "\n"
        file.write(sentence)





