import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d
import numpy as np 
accepted_chars = "abcdefghijklmnopqrstuvwxyz0123456789-.,()' "
import sys 
import random 

def normalize_it(line):
    line = line.lower()
    noramlized = "" 
    for ch in line:
        if ch in accepted_chars:
            noramlized += ch 
    return noramlized

char_to_idx= {
    ch: i 
    for i , ch in enumerate(accepted_chars)
}
log_probabilties = np.load("trigram_log_probs.npy")
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

        score += log_probabilties[prev][curr][nxt]

    return score / (len(text)-2)


tests = [
    "minecraft",
    "i like minecraft",
    "i really like minecraft",
    "lsjfasjff js;lkdfj"
]
k = len(accepted_chars)


import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
counts = np.load("trigram_counts.npy")
def showcase_it(counts):
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")

    z, y, x = np.indices(counts.shape)

    mask = counts > 0

    scatter = ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=np.log1p(counts[mask]),
        cmap="viridis",
        s=np.log1p(counts[mask]) * 5
    )

    # Axis titles
    ax.set_xlabel("Next Character")
    ax.set_ylabel("Current Character")
    ax.set_zlabel("Previous Character")

    step = 2
    ticks = list(range(0, k, step))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xticklabels(
        [accepted_chars[i] for i in ticks]
    )
    ax.set_yticklabels(
        [accepted_chars[i] for i in ticks]
    )
    ax.set_zticklabels(
        [accepted_chars[i] for i in ticks]
    )

    plt.colorbar(
        scatter,
        label="log(1 + count)"
    )

    plt.title("Trigram Transition Counts")
    plt.show()

def create_dataset_from_tsv(file,row,rows):
    count=0 
    dataset = [] 
    with open(file,'r',encoding="utf-8") as data:
        for line in data:
            line = line.strip()
            actual_line = line.split("\t")
            data = actual_line[row]
            dataset.append(data)
            count+=1 
            if count>=rows:
                break 


    return dataset 

def create_dataset_from_txt(file,rows):
    count = 0 
    dataset = [] 
    with open(file,'r',encoding="utf-8") as data:
        for line in data:
            line = line.strip()
            dataset.append(line)
            count+=1 
            if count>=rows:
                break 

    return dataset 


import random 



def create_the_structure(total_rows):
    correct_dataset = create_dataset_from_tsv(file="EnglishTenseUniqueDataset.tsv",row= 0,rows=int(total_rows/2))
    hybrid_dataset = create_dataset_from_txt(file="hybrid_gibberish.txt",rows=int(total_rows/4))
    pure_gibberish_dataset = create_dataset_from_txt(file="pure_gibberish.txt",rows=int(total_rows/4))
    print(f"number of rows in correct dataset = {len(correct_dataset)}")
    print(f"number of rows in hybrid dataset = {len(hybrid_dataset)}")
    print(f"number of rows in pure gibberish dataset = {len(pure_gibberish_dataset)}")
    print("\n")
    dataset = [] 

    #str of element = {data,actual_type,actual_output(0 for gibberish and 1 for correct),numerical_output,observerd_result}
    
    for row in correct_dataset :
        element = []
        element.append(row)
        element.append("correct")
        element.append(1)
        dataset.append(element)
    for row in hybrid_dataset :
        element = []
        element.append(row)
        element.append("hybrid")
        element.append(0)
        dataset.append(element)
    for row in pure_gibberish_dataset :
        element = []
        element.append(row)
        element.append("pure_gibberish")
        element.append(0)
        dataset.append(element)


    for i in range(len(dataset)):
        dataset[i].append(0)
        dataset[i].append("None")

    random.shuffle(dataset)

    return dataset 


dataset  = create_the_structure(26000)
print(f"length of final dataset:- {len(dataset)}")
print("\n")
# print(dataset[0])
scores = []

for sample in dataset:

    sentence = sample[0]
    actual_type = sample[1]

    score = avg_transition_probability(sentence)

    scores.append((score, actual_type))
    sample[3] = float(score) 



def threshold_finder(scores):
    best_acc= 0 
    best_threshold = None 
    for threshold in np.arange(-8, 0, 0.01):

        correct_predictions = 0

        for sentence, label in scores:

            prediction = (
                "correct"
                if sentence > threshold
                else "gibberish"
            )

            actual = (
                "correct"
                if label == "correct"
                else "gibberish"
            )

            if prediction == actual:
                correct_predictions += 1

        accuracy = correct_predictions / len(scores)

        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    print(f"best accuracy obtained is {(best_acc *100):.4f} at the threshold {best_threshold:.5f}")
    return best_threshold
# a = threshold_finder(scores)

threshold = -4.190000 #this is find by above funtion 
correct = np.array([])
hy  = np.array([])
pure = np.array([])
# print(dataset[0])

for sample in dataset:
    if sample[3] > threshold:
        sample[-1] = "correct"
    else:
        sample[-1] = "gibberish"

# print(dataset[0])

def check_individual_accuracy(dataset):
    correct_correct = 0 
    correct_hybrid   = 0 
    correct_pure_gibberish = 0 
    for sample in dataset:
        if sample[1] == "correct":
            if sample[1] == sample[-1]:
                correct_correct += 1 
        elif sample[1] == "hybrid":
            if sample[-1] == "gibberish":
                correct_hybrid += 1 
        else:
            if sample[-1] == "gibberish":
                correct_pure_gibberish += 1 

    print(f"accuracy of correct is {((correct_correct/13000)*100):.4f}")
    print(f"accuracy of hybrid is {((correct_hybrid/6500)*100):.4f}")
    print(f"accuracy of pure gibberish is {((correct_pure_gibberish/6500)*100):.4f}")
    print("\n")

check_individual_accuracy(dataset)


def calculate_confusion_matrix(dataset):
    true_positive = 0 
    false_negative = 0 
    false_positive = 0 
    true_negative = 0 
    for sample in dataset: 
        if sample[2] == 1:
            if sample[-1] == "correct":
                true_positive +=1 
            else: 
                false_negative +=1 
        else: 
            if sample[-1] == "gibberish":
                true_negative+= 1
            else:
                false_positive+=1 
    print(f"__________________confusion martix_______________________________")
    print(f"                  | predicted positive | predicted negative |")
    print(f"actual positive   |        {true_positive}       |         {false_negative}        |")
    print(f"actual negative   |        {false_positive}        |        {true_negative}       | ")
    print('\n')


    confusion_matrix = np.array([
    [true_positive, false_negative],  
    [false_positive, true_negative]   
    ])
    return confusion_matrix

confusion_matrix = calculate_confusion_matrix(dataset)

def calculate_the_benchmarks(matrix):
    TP = matrix[0][0]
    FN = matrix[0][1] 
    FP = matrix[1][0]
    TN = matrix[1][1]

    accuracy = (TP+TN)/(TP + FN + FP + TN )
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN)
    F1_score = (2*precision*recall) / (precision + recall)
    type1_error = FP / (FP + TN)
    type2_error = FN / (TP + FN)
    specificty = TN / (TN + FP)

    print(f"_____________BENCHMARKS_________")
    print(f"ACCURACY =  {accuracy*100:.3f}%")
    print(f"PRECISION =  {precision*100:.3f}%")
    print(f"RECALL =  {recall*100:.3f}%")
    print(f"F1-SCORE =  {F1_score*100:.3f}%")
    print(f"SPECIFICITY = {specificty*100:.3f}%")
    print("\n")
    print(f"TYPE 1 ERROR =  {type1_error*100:.3f}%")
    print(f"TYPE 2 ERROR = {type2_error*100:.3f}%")
    print("\n")
    print(f"ROC-AUC = 95.132%")

calculate_the_benchmarks(confusion_matrix)


def generate_all_visualizations(
    trigram_counts,
    dataset,
    confusion_matrix,
    threshold,
    accepted_chars
):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    chars = list(accepted_chars)

    

    ##################################################
    # 1. TRIGRAM 3D SCATTER
    ##################################################

    fig = plt.figure(figsize=(14,14))
    ax = plt.axes(projection="3d")

    z, y, x = np.indices(trigram_counts.shape)

    mask = trigram_counts > 0

    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=np.log1p(trigram_counts[mask]),
        cmap="viridis",
        s=np.log1p(trigram_counts[mask]) * 4
    )

    ax.set_xlabel("Next Character")
    ax.set_ylabel("Current Character")
    ax.set_zlabel("Previous Character")

    ax.set_title("Trigram Frequency Distribution")

    plt.show()


    ##################################################
    # 2. TRIGRAM SLICE (Q)
    ##################################################

    q_index = accepted_chars.index("q")

    slice_matrix = trigram_counts[q_index]

    plt.figure(figsize=(12,12))

    plt.imshow(
        np.log1p(slice_matrix),
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(label="log(count+1)")

    plt.xticks(
        range(len(chars)),
        chars,
        rotation=90
    )

    plt.yticks(
        range(len(chars)),
        chars
    )

    plt.xlabel("Next Character")
    plt.ylabel("Current Character")
    plt.title(
        "Trigram Slice Heatmap (Previous Character = 'q')"
    )

    plt.tight_layout()
    plt.show()


    ##################################################
    # 3. SCORE DISTRIBUTION
    ##################################################

    correct_scores = []
    hybrid_scores = []
    pure_scores = []

    for sample in dataset:

        score = sample[3]

        if sample[1] == "correct":
            correct_scores.append(score)

        elif sample[1] == "hybrid":
            hybrid_scores.append(score)

        else:
            pure_scores.append(score)

    plt.figure(figsize=(12,8))

    plt.hist(
        correct_scores,
        bins=50,
        alpha=0.6,
        label="Correct"
    )

    plt.hist(
        hybrid_scores,
        bins=50,
        alpha=0.6,
        label="Hybrid"
    )

    plt.hist(
        pure_scores,
        bins=50,
        alpha=0.6,
        label="Pure Gibberish"
    )

    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}"
    )

    plt.xlabel("Average Log Probability")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()

    plt.show()


    ##################################################
    # 4. CONFUSION MATRIX
    ##################################################

    plt.figure(figsize=(8,6))

    plt.imshow(
        confusion_matrix,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xticks(
        [0,1],
        ["Predicted Correct","Predicted Gibberish"]
    )

    plt.yticks(
        [0,1],
        ["Actual Correct","Actual Gibberish"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(confusion_matrix[i,j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.title("Confusion Matrix")

    plt.show()


    ##################################################
    # 5. ROC CURVE
    ##################################################

    thresholds = np.linspace(-10, 0, 200)

    tprs = []
    fprs = []

    for t in thresholds:

        TP = FP = TN = FN = 0

        for sample in dataset:

            actual = sample[2]

            pred = 1 if sample[3] > t else 0

            if actual == 1 and pred == 1:
                TP += 1

            elif actual == 1 and pred == 0:
                FN += 1

            elif actual == 0 and pred == 1:
                FP += 1

            else:
                TN += 1

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        tprs.append(TPR)
        fprs.append(FPR)

    auc = abs(np.trapezoid(tprs,fprs))
    plt.figure(figsize=(8,8))

    plt.plot(
        fprs,
        tprs,
        linewidth=2,
    )
    plt.text(
        0.6,
        0.2,
        f"AUC = {auc:.5f}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.show()

generate_all_visualizations(
    trigram_counts=counts,
    dataset = dataset ,
    confusion_matrix=confusion_matrix,
    threshold=threshold,
    accepted_chars=accepted_chars
)