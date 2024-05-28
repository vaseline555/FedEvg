import torch
import logging
import medmnist

logger = logging.getLogger(__name__)



# helper method to fetch dataset from `torchvision.datasets`
def fetch_medmnist(args, dataset_name, root, transforms):
    logger.info(f'[LOAD] [{dataset_name.upper()}] Fetching dataset!')
    
    # check support of size
    assert args.resize in [28, 64, 128, 224], f"Size of {args.resize} is not supported...!\
        (please select among [28, 64, 128, 224])"
        
    # get config of dataset
    config = medmnist.INFO[dataset_name.lower()]

    # parse datasets
    raw_train = getattr(medmnist, config['python_class'])(
        split='train', transform=transforms[0], download=True, size=args.resize
    )
    raw_test = getattr(medmnist, config['python_class'])(
        split='test', transform=transforms[1], download=True, size=args.resize
    )

    # flatten label array: (B, 1) -> (B,)
    raw_train.labels = raw_train.labels.squeeze()
    raw_test.labels = raw_test.labels.squeeze()
    
    # for consistency
    setattr(raw_train, 'targets', torch.tensor(raw_train.labels).view(-1))
    setattr(raw_test, 'targets', torch.tensor(raw_test.labels).view(-1))

    # adjust arguments
    args.in_channels = config['n_channels']
    args.num_classes = len(config['label'])

    logger.info(f'[LOAD] [{dataset_name.upper()}] ...fetched dataset!')
    return raw_train, raw_test, args
