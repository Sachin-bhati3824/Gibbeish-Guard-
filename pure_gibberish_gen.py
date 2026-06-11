import random

def generate_realistic_gibberish(num_sentences=10):
    accepted_chars = "abcdefghijklmnopqrstuvwxyz0123456789-.,()' "
    # Keyboard rows to simulate real human mashing
    keyboard_mashes = ["asdf", "hjkl", "qwerty", "zxcv", "mnbvc"]
    dataset = []

    for _ in range(num_sentences):
        sentence_words = []
        len_of_sentence = random.randint(3, 8)
        
        for _ in range(len_of_sentence):
            # Mix up the types of gibberish so the model learns edge cases
            gibberish_type = random.choice(['random', 'mash', 'repeat'])
            
            if gibberish_type == 'random':
                # Pure random string (Your original logic)
                word = ''.join(random.choice(accepted_chars[:-1]) for _ in range(random.randint(3, 10)))
            
            elif gibberish_type == 'mash':
                # Simulates striking adjacent keys (e.g., "asdfghasdf")
                base = random.choice(keyboard_mashes)
                word = ''.join(random.choice(base) for _ in range(random.randint(5, 12)))
                
            elif gibberish_type == 'repeat':
                # Simulates held-down keys (e.g., "g0000000d")
                char = random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                word = char * random.randint(6, 15)
                
            sentence_words.append(word)
            
        dataset.append(" ".join(sentence_words))
    return dataset
with open("pure_gibberish.txt","w",encoding="utf-8") as file :
    dataset=  generate_realistic_gibberish(6500)
    for i in range(len(dataset)):
        file.write(dataset[i]+"\n")
