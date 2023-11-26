import copy
from tqdm import tqdm

import torch 

def train(epoch, model, loader, optimizer, logger):
    model.train()

    count = 0
    loss_sum = 0
    
    with tqdm(total=len(loader), ncols=90) as pbar:
        for x, label in loader:
            # clear grad
            optimizer.zero_grad()

            # feed forward
            x = x.to(dtype=torch.float32, device='cuda')
            label = label.to(dtype=torch.int64, device='cuda')
            
            loss = model(x, label)
            
            # backpropagation
            loss.backward()
            optimizer.step()

            # log
            count += 1
            loss_sum += loss.item()
            if len(loader) * 0.02 <= count:
                logger.log_metric('Loss', loss_sum / count)
                count = 0
                loss_sum = 0
            
            # pbar
            desc = f'[{epoch}]|(loss): {loss.item():.3f}'
            pbar.set_description(desc)
            pbar.update(1)

def test_inference(DB, model, loader):
    # set test mode
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    
    corrects = [0 for _ in range(len(DB.genre_dict.keys()))]
    total_samples = [0 for _ in range(len(DB.genre_dict.keys()))]
    predicts = [0 for _ in range(len(DB.genre_dict.keys()))]
    
    with torch.set_grad_enabled(False):
        for x, label in tqdm(loader, desc='test', ncols=90):                
            # to cuda
            x = x.to(dtype=torch.float32, device='cuda')
            
            # inference
            x = model(x)
            p = softmax(x)
            p = torch.max(p, dim=1)[1]
            
            # count
            for i in range(p.size(0)):
                _p = p[i].item()
                l = label[i].item()
                predicts[_p] += 1
                total_samples[l] += 1
                if _p == l:
                    corrects[l] += 1
                
    return corrects, predicts, total_samples