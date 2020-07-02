from sari.SARI import SARIsent
import numpy as np


def sari_score(inputs, predicted, reference):
    SARI_score = []
    if len(test_complexes) == len(test_simples) == len(test_simples):
        for line in range(len(test_complexes)):
            SARI = SARIsent(inputs[line], predicted[line], reference[line])
            SARI_score.append(SARI)
        SARI_score_total = np.average(SARI_score)
    else:
        print("SARI: The number of Predicted and their references should be the same...")
    return SARI_score_total

total_sari_score = sari_score(test_complexes, test_simples, test_simples)
print("SARI Score is : {}".format(total_sari_score))


#r S = β S ARI (X, Ŷ , Y ) + (1 − β) S ARI (X, Y, Ŷ )
beta = 1.0
def simplicity(inputs, predicted, reference):
    sari = sari_score(inputs, predicted, reference)
    sari_reverse = sari_score(inputs, reference, predicted)
    return (beta * sari) + ((1.0 -beta) * sari_reverse)
r_s = simplicity(test_complexes, test_simples, test_simples)
