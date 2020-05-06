config = {
    'device': 'cuda:0',
    'seed': 42,

    # config for logging
    'logging': {
        'log_file': 'run.log',
        'fmt': '%(asctime)s: %(message)s',
        'level': 'INFO',
    },

    # config to load and save network
    'net': {
        'saved_net_path': None,
        'net_path': 'models/resnet.py',
        'saved_params_path': None
    },

    # dataset path
    'dataset': {
        'train': '../data/train',
        'val': '../data/val',
        'test': '../data/test',
        'train_size': 3180066,
        'val_size': 97128,
        'test_size': 96737
    },

    'model_cur': 'MLP',
    'GMF_model': '../train_results/GMF.pth',
    'MLP_model': '../train_results/MLP.pth',
    'NMF_model': '../train_results/NMF.pth',
    'NMF-pre_model': '../train_results/NMF-pre.pth'
}