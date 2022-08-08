def preprocessing(path):
    """
    The function returns preprocessed dataset, 
    taking path to dataset as input
    """
    with open(path, "r") as f:
        dataset = f.readlines()
    for instance in dataset:
        instance=list(map(int, instance[:-1].split(",")))
    return dataset

if __name__=="__main__":
    a=preprocessing("optdigits.tra")
    print(a[0])

