import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import update_parameters, get_accuracy,addvalue
import numpy as np
def maml_operation(args,sec,dic):
    print(sec)
    dataloader = dic['dataloader']
    model = dic['model']
    meta_optimizer = dic['optimizer']
    lossf=dic['lossf']
    accf=dic['accf']
    writer=dic['writer']

    for idx ,batch in enumerate(dataloader):
        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']
        for train_input,train_target,test_input,test_target in zip(train_inputs,train_targets,test_inputs,test_targets):
            model.zero_grad()
            train_input = train_input.to(device=args.device)
            train_target = train_target.to(device=args.device)
            test_input = test_input.to(device=args.device)
            test_target = test_target.to(device=args.device)

            train_logit = model(train_input)
            inner_loss = lossf(train_logit, train_target)
            model.zero_grad()
            params = update_parameters(model, inner_loss,
                                       step_size=args.step_size, first_order=args.first_order)
            test_logit = model(test_input, params=params)
            outer_loss += lossf(test_logit, test_target)
            with torch.no_grad():
                accuracy += accf(test_logit, test_target)
        outer_loss.div_(args.batch_size)
        accuracy.div_(args.batch_size)
        print(f'idx:{idx},acc:{accuracy:.4f},loss:{outer_loss.item():.4f}')
        addvalue(writer,'loss:train',outer_loss.item())
        if sec=='train':
            outer_loss.backward()
            meta_optimizer.step()
        if idx > args.num_batches:
            break

def own_test(args,sec,dic):
    print(sec)
    assert sec=='test'
    dataloader = dic['dataloader']
    model = dic['model']
    meta_optimizer = dic['optimizer']
    lossf=dic['lossf']
    accf=dic['accf']
    writer=dic['writer']

    batch=dataloader.dataset.train()
    model.zero_grad()
    train_input, train_target = batch['train']
    train_input = train_input.to(device=args.device)
    train_target = train_target.to(device=args.device)

    train_logit = model(train_input)
    loss = lossf(train_logit, train_target)
    loss.backward()
    meta_optimizer.step()
    print(f'test:loss:{loss.item():.4f}')
    addvalue(writer,'loss:test',loss.item())

    accuracy = 0
    test_loss=[]
    #num dataset.test size
    with torch.no_grad():
        for idx,(raw,mask) in enumerate(dataloader):
            raw=raw.to(args.device)
            mask=mask.to(args.device)
            output=model(raw)
            test_loss.append(lossf(output,mask).item())

    print(f'acc:test {1-np.mean(test_loss):.4f}')
    addvalue(writer,'acc:test',1-np.mean(test_loss))

