import pickle 
i = 0 
english_words = [] 
with open("vocabulary.pkl", "rb") as f:
    english_words = pickle.load(f)
print(len(english_words))