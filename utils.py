def get_device(model):
    return next(model.parameters()).device

def train(dl, loss_fn, model, optimizer):
    device = get_device(model)
    model.train()
    mean_loss = 0.
    for inp, tgt in dl:
        optimizer.zero_grad()
        out = model(inp.to(device))
        loss = loss_fn(out, tgt.to(device))
        loss.backward()
        optimizer.step()
        mean_loss += loss.item() / len(dl)
    return mean_loss