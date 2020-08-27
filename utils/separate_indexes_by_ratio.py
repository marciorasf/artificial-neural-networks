def separateIndexesByRatio(nSamples, ratio):
    step = 1 / ratio
    selectedIndexes = []
    nonSelectedIndexes = []
    for index in range(nSamples):
        if index % step < 1:
            selectedIndexes.append(index)
        else:
            nonSelectedIndexes.append(index)

    return selectedIndexes, nonSelectedIndexes

