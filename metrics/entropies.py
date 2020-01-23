import numpy as np
from  scipy.stats import entropy

def bet_entropy(orthop, n_classes):
    probs = np.zeros([n_classes])
    for elem in orthop:
        for cl in elem:
            probs[cl] += 1.0/len(elem)
    probs /= sum(probs)
    print(probs)
    return entropy(probs, base=2)
	
def sample_dataset(ys):
    y_res = np.zeros([len(ys)])
    i = 0
    y_res = [[np.random.choice(y)] for y in ys]
    return y_res

def avg_entropy(orthop, n_classes, samples):
    entrop = 0.0
    for i in range(samples):
        sample = sample_dataset(orthop)
        entrop += bet_entropy(sample,n_classes)
    entrop /= samples
    return entrop