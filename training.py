import torch
import numpy as np
import editdistance
import matplotlib.pyplot as plt
import tqdm

def train(model, optimizer, train_loader, state):
    """
    model: an instance of Seq2Seq
    optimzer: torch.optimizer module
    train_loader
    """
    epoch, n_epochs = state

    losses = []
    cers = []

    t = tqdm.tqdm(train_loader)

    ## Set the self.training attribute in Seq2Seq module to True
    model.train()

    for batch in t:
        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format(epoch, n_epochs, model.training))
        loss, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    avg_loss = np.mean(losses)

    return model, optimizer,avg_loss

def evaluate(model, eval_loader,targIdx):

    losses = []
    accs = []

    t = tqdm.tqdm(eval_loader)
    model.eval()

    with torch.no_grad():
        for batch in t:
            t.set_description(" Evaluating... (train={})".format(model.training))
            loss, logits, labels , sentence_pred = model.loss(batch)
            preds = logits.detach().cpu().numpy()

            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            ## This is where I need to work on plugging in Bag of Words/ BLEU Score / in this repo, it uses Levenshtein distance to measure how similar two strings are, but to do that, I need to translate the phrases first, I will use that next
            acc = 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)
            losses.append(loss.item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{}'.format(np.mean(losses)))
            t.update()

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig("data/att.png")
    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))
    
    avg_loss = np.mean(losses)
    accuracy = None

    return avg_loss,accuracy


 
