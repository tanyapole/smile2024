class PartialDs:
    def __init__(self, ds, idxs):
        self.ds = ds
        self.idxs = idxs
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, i): 
        return self.ds[self.idxs[i]]


def compute_entropies(out):
    prob = out.detach().cpu().softmax(dim=1)
    assert ((prob.sum(dim=1) - 1).abs() < 1e-5).all()
    entropy = - prob * prob.log()
    entropy = entropy.sum(dim=1) # sum over classes
    entropy = entropy.mean(dim=(-1,-2)) # average over pixels
    return entropy