import torch

from collections import OrderedDict
import numpy as np
def update_parameters(model, loss, step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.

    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.

    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order)

    params = OrderedDict()
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        params[name] = param - step_size * grad

    return params

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())
import pickle
import matplotlib.pyplot as plt
def savedic(writer,fol='data'):
    n=1
    dic={}
    fig=plt.figure()
    axdic={}
    for tag,value in writer:
        dic.setdefault(tag.split(':')[0],{}).setdefault(tag.split(':')[1],[]).append(value)
    maxlen=np.max(np.max([[len(dic[upkey][key]) for key in  dic[upkey]] for upkey in dic]))
    minlen=np.min(np.min([[len(dic[upkey][key]) for key in  dic[upkey]] for upkey in dic]))
    maxlen=minlen*maxlen//minlen
    for upkey in dic:
        axdic[upkey]=fig.add_subplot(len(dic),1,n)
        n+=1
        for key in dic[upkey]:
            y=dic[upkey][key][:maxlen]
            x=range(0,maxlen,maxlen//len(y))[:len(y)]
            axdic[upkey].plot(x, y, label=f'{upkey}:{key}')
    for key in axdic:
        axdic[key].legend()
    fig.savefig(f'{fol}/graphs.png')
    with open(f'{fol}/data.pkl','wb') as f:
        pickle.dump(dict,f)

def save(e,model,savedpath,dic=None):
    if dic:
        savedic(dic,savedpath)
    torch.save(model.state_dict(), savedpath+'/model.pth')
    with open(f'{savedpath}/.epoch','w') as f:
        f.write(f'{e}')

def addvalue(writer,tag,value):
    writer.append((tag,value))
