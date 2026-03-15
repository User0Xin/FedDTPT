def accuracy(preds, labels):

    correct = 0

    for p,l in zip(preds,labels):

        if p == l:
            correct += 1

    return correct/len(labels)