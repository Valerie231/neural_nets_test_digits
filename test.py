def preprocessing(path):
    """
    The function returns preprocessed dataset, 
    taking path to dataset as input
    """
    dataset=[]
    with open(path, "r") as f:
        dataset_raw = f.readlines()
    for instance in dataset_raw:
        instance=list(map(int, instance[:-1].split(",")))
        instance[:-1]=[el/16 for el in instance[:-1]]
        dataset.append(instance)
    return dataset

if __name__=="__main__":
    a=preprocessing("optdigits.tra")
    print(a[0])

