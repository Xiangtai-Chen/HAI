import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import nni

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
# RECEIVED_PARAMS = nni.get_next_parameter()
# lr = RECEIVED_PARAMS['learning_rate']
# dr = RECEIVED_PARAMS['dropout_rate']
lr = 0.003
dr = 0.6
default_configure = {
    'lr': lr,             # Learning rate
    'num_heads': [2],        # Number of attention heads for node-level attention
    'hidden_units': 32,
    'dropout': dr,
    'weight_decay': 0.001,
    'num_epochs': 10,
    'patience': 20
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'dblp' 
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_dblp(remove_self_loop):
    NODE_FEATURE = 'Dataprocessing/dblp.v12/data/node_feature.pkl'
    EDGES = 'Dataprocessing/dblp.v12/data/edges.pkl'
    with open(NODE_FEATURE,'rb') as f:
        node_features = pickle.load(f)
    with open(EDGES,'rb') as f:
        edges = pickle.load(f)
    # A_ap,A_pa,A_ao,A_oa
    a_vs_p = edges[0]
    p_vs_a = edges[1]
    a_vs_o = edges[2]
    o_vs_a = edges[3]

    hg = dgl.heterograph({
        ('author', 'ap', 'paper'): a_vs_p.nonzero(),
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ao', 'org'): a_vs_o.nonzero(),
        ('org', 'oa', 'author'): o_vs_a.nonzero()
    })

    features = torch.tensor(node_features, dtype=torch.float32)
    num_classes = 64
    return hg, features, num_classes

def load_data(dataset, remove_self_loop=False):
    if dataset == 'dblp':
        return load_dblp(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))