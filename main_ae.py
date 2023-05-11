from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
import utils
from ae_datasets import d_selector, Dataloader
import models
import losess
import eval
import yaml
import shutil


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help='Configuration file with hyperparameters.')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)

plots = config.get('plots', 0)
exp_name = config.get('exp_name', 'exp_1')
seed = config.get('seed', 1)
outf = config.get('outf', 'outputs_ae')

train = config['train']
epochs = train.get('epochs', 100)
no_cuda = train.get('no-cuda', 0)
log_interval = train.get('log_interval', 100)
test_interval = train.get('test_interval', 2)
generate_interval = train.get('generate-interval', 100)
lr = float(train.get('lr', 1e-4))

data = config['data']
dataset_name = data.get('dataset', 'community_ours')
with_pos = data.get('with_pos', 1)
n_nodes = data.get('n_nodes', 10)
  
model_dict = config['model']
model = model_dict.get('model', 'ae_egnn')
nf = model_dict.get('nf', 64)
emb_nf = model_dict.get('emb_nf', 8)
K = model_dict.get('K', 2)
attention = model_dict.get('attention', 0)
noise_dim = model_dict.get('noise_dim', 0)
n_layers = model_dict.get('n_layers', 4)
reg = float(model_dict.get('reg', 1e-3))
clamp = model_dict.get('clamp', 1)
weight_decay = float(model_dict.get('weight_decay', 1e-16))



use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

utils.create_folders(outf, exp_name)

if with_pos:
    print('=======')
    print('Using coordinates')
    print('=======')

#
dataset = d_selector.retrieve_dataset(dataset_name, with_pos=with_pos, K=K, partition="train", directed=True, n_nodes=n_nodes)
train_loader = Dataloader(dataset, batch_size=1)
dataset = d_selector.retrieve_dataset(dataset_name, with_pos=with_pos, K=K, partition="val", directed=True, n_nodes=n_nodes)
val_loader = Dataloader(dataset, batch_size=1, shuffle=False)
dataset = d_selector.retrieve_dataset(dataset_name, with_pos=with_pos, K=K, partition="test", directed=True, n_nodes=n_nodes)
test_loader = Dataloader(dataset, batch_size=1, shuffle=False)

if model == 'ae':
    model = models.AE(hidden_nf=nf, embedding_nf=emb_nf, noise_dim=noise_dim, act_fn=nn.SiLU(),
                      learnable_dec=1, device=device, attention=attention, n_layers=n_layers)
elif model == 'ae_rf':
    model = models.AE_rf(embedding_nf=K, nf=nf, device=device, n_layers=n_layers, reg=reg,
                         act_fn=nn.SiLU(), clamp=clamp)
elif model == 'ae_egnn':
    model = models.AE_EGNN(hidden_nf=nf, K=K, act_fn=nn.SiLU(), device=device, n_layers=n_layers,
                           reg=reg, clamp=clamp)
elif model == 'baseline':
    model = models.Baseline(device=device)
else:
    raise Exception('Wrong model %s' % model)

print(model)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

pr = eval.ProgressReporter(path=outf + '/' + exp_name, file_name='/output.json')


def checkpoint(model_path):
    torch.save(model.state_dict(), model_path)


def train(epoch, loader):
    lr_scheduler.step(epoch)
    model.train()
    res = {'epoch': epoch, 'loss': 0, 'bce': 0, 'kl': 0, 'kl_coords': 0, 'adj_err': 0, 'coord_reg': 0, 'counter': 0, 'wrong_edges': 0, 'gt_edges': 0, 'possible_edges': 0}
    magnitudes = {'value':0, 'counter':0}
    for batch_idx, data in enumerate(loader):
        graph = data[0]

        if with_pos:
            coords = graph.get_coords()
            coords = coords.to(device)
        else:
            coords = None

        nodes, edges, edge_attr, adj_gt = graph.get_dense_graph(store=True, loops=False)
        nodes, edges, edge_attr, adj_gt = nodes.to(device), [edges[0].to(device), edges[1].to(device)], edge_attr.to(device), adj_gt.to(device).detach()
        n_nodes = nodes.size(0)
        optimizer.zero_grad()

        adj_pred, z = model(nodes, edges, coords, edge_attr)
        bce, kl = losess.vae_loss(adj_pred, adj_gt, None, None)
        kl_coords = torch.zeros(1)
        loss = bce

        loss = loss # normalize loss by the number of nodes

        loss.backward()
        optimizer.step()

        res['loss'] += loss.item()
        res['bce'] += bce.item()
        res['kl'] += kl.item()
        res['kl_coords'] += kl_coords.item()
        wrong_edges, adj_err = eval.adjacency_error(adj_pred, adj_gt)

        res['adj_err'] += adj_err
        res['counter'] += 1
        res['wrong_edges'] += wrong_edges
        res['gt_edges'] += torch.sum(adj_gt).item()
        res['possible_edges'] += n_nodes ** 2 - n_nodes
        if batch_idx % log_interval == 0:
            print('===> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        magnitudes['value'] += torch.mean(torch.abs(z))
        magnitudes['counter'] += 1
    error = res['wrong_edges'] / res['possible_edges']
    print('Train avg bce: %.4f \t KL %.4f \t KL_coords %.4f \tAdj_err %.4f \nWrong edges %d \t gt edges %d \t Possible edges %d \t Error %.4f' % (res['bce'] / res['counter'], res['kl'] / res['counter'], res['kl_coords'] / res['counter'], res['adj_err'] / res['counter'], res['wrong_edges'], res['gt_edges'], res['possible_edges'], error))


def test(epoch, loader):
    model.eval()
    res = {'epoch': epoch, 'loss': 0, 'bce': 0, 'kl': 0, 'kl_coords': 0, 'adj_err': 0, 'counter': 0, 'wrong_edges': 0, 'gt_edges': 0, 'possible_edges': 0, 'tp': 0, 'fp': 0, 'fn': 0}
    with torch.no_grad():
        for idx, data in enumerate(loader):
            graph = data[0]
            n_nodes = graph.get_num_nodes()
            
            if with_pos:
                coords = graph.get_coords()
                coords = coords.to(device)
            else:
                coords = None

            nodes, edges, edge_attr, adj_gt = graph.get_dense_graph(store=True, loops=False)
            nodes, edges, edge_attr, adj_gt = nodes.to(device), [edges[0].to(device), edges[1].to(device)], edge_attr.to(device), adj_gt.to(device)

            adj_pred, mu = model(nodes, edges, coords, edge_attr)
            bce, kl = losess.vae_loss(adj_pred, adj_gt, None, None)
            loss = bce

            res['loss'] += loss.item()
            res['bce'] += bce.item()
            res['kl'] += kl.item()
            tp, fp, fn = eval.tp_fp_fn(adj_pred, adj_gt)
            res['tp'] += tp
            res['fp'] += fp
            res['fn'] += fn
            wrong_edges, adj_err = eval.adjacency_error(adj_pred, adj_gt)
            res['adj_err'] += adj_err
            res['counter'] += 1
            res['wrong_edges'] += wrong_edges
            res['gt_edges'] += torch.sum(adj_gt).item()
            res['possible_edges'] += n_nodes ** 2 - n_nodes

    res = utils.normalize_res(res, keys=['loss', 'bce', 'kl', 'kl_coords', 'adj_err'])
    error = res['wrong_edges']/ res['possible_edges']
    f1_score = 1.0*res['tp'] / (res['tp'] + 0.5*(res['fp'] + res['fn']))
    print('Test on %s \t \t \t \t loss: %.4f \t  bce: %.4f \t  kl: %.4f \t  kl_coords: %.4f \t Adj_err %.4f \nWrong edges %d \t gt edges %d \t Possible edges %d \t Error %.4f \t TP: %d \t FP: %d \t FN: %d \t F1-score: %.4f' % (loader.dataset.partition,
                                                                                    res['loss'],
                                                                                    res['bce'],
                                                                                    res['kl'],
                                                                                    res['kl_coords'],
                                                                                    res['adj_err'],
                                                                                    res['wrong_edges'],
                                                                                    res['gt_edges'],
                                                                                    res['possible_edges'],
                                                                                     error, res['tp'], res['fp'], res['fn'], f1_score))
    pr.add_epoch(res, loader.dataset.partition)
    return res


if __name__ == "__main__":

    best_bce_val = 1e8
    best_res_test = None
    best_epoch = 0

    # Copy config file to keep track of the experiment
    new_path = os.path.join(outf, exp_name, 'config.yaml')
    shutil.copy(args.config, new_path)

    for epoch in range(0, epochs):
        train(epoch, train_loader)
        if epoch % test_interval == 0:
            res_train = test(epoch, train_loader)
            res_val = test(epoch, val_loader)
            res_test = test(epoch, test_loader)

            if res_val['bce'] < best_bce_val:
                best_bce_val = res_val['bce']
                best_res_test = res_test
                best_epoch = epoch
            print("###############\n### Best result is: bce: %.4f, wrong_edges %d, error: %.4f, epoch %d" % (best_res_test['bce'],
                                                                                     best_res_test['wrong_edges'],
                                                                                     best_res_test['wrong_edges']/best_res_test['possible_edges'],
                                                                                     best_epoch))
            print("###############")

    model_path = os.path.join(outf, exp_name, 'model.pt')
    checkpoint(model_path)


