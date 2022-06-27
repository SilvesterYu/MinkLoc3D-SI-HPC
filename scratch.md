# Clues

## training/train.py
Usage
```
python train.py --config ../config/config_usyd.txt --model_config ../models/minkloc_config.txt
```

**params**: parameters cooresponding to the specified config and model_config, returned by MinkLocParams class in misc/utils.py (a class for fetching specific params from the config filess)

**dataloaders**: dataloaders according to **params**, returned from make_dataloaders() in datasets/dataset_utils.py.

**do_train()**: called from training/trainer.py. The main function for the training process.

---

## training/trainer.py

**do_train()** defined here. 

(0) **momdel**: a model returned from model_factory() from models/model_factory.py according to params. It is a model defined in class MinkLoc from models/minkloc.py

(1) It first gets from the params the model name, then creates a path **model_pathname** to be the path where the weights are stored. 

(2) checks for cuda device. if available run on cuda. if not device is set to "cpu"

(3) **loss_fn**: the specific loss function to use. Returned from **make_loss()** in models/loss.py. This function returns the loss function corresponding to what is specified in params.

(4) **ooptimiaer**: an optimizer (Adam) with weight decay (or none) specified in params

(5) **scheduler**: according to params.scheduler.

(6) Initialize log directory and SummaryWriter taking the date time right now. So training logs can be stored with date time in their file names.

(7) **phases**: either ['train', 'val'] or just ['train'] according to params.

(8) (for1) There is a big for loop iterating over the epochs. in each epoch, a smaller for loop (for2) iterates over the phases ('train' and perhaps'val'). If train, then call model.train(). If eval, then call model.eval(). (These are inherited functions from torch.nn.Module)

(9) (for3) Iterate through batch and masks in dataloader (as in the comments) # batch is (batch_size, n_points, 3) tensor # labels is list with indexes of elements forming a batch

(10) count positive and negative examples, skip batch without positives or negatives.

(11) [problem] if visualize only does "pass". visualization is commented out.

(12) locally enable/disable gradient computation. [question] embeddings = model(batch) # Compute embeddings of all elements. How exactly does batch work in model

(13) compute loss by putting embeddings, positive mask, and negative mask into loss function

(14) calls loss.backward() when the phase is "train". loss is returned by loss_fn, which is returned by make_loss(params) from models/loss.py. it will either return BatchHardTripletLossWithMasks or BatchHardContrastiveLossWithMasks from the same file. loss returned by BatchHardTripletLossWithMasks is a tensor. this is eventually captured by the loss above. It is a torch.Tensor, backward() is a torch.Tensor function. It computes the gradient of current tensor w.r.t. graph leaves. [question] will look into this more.

(14.1) Adam optimiaer is also used here. (torch.optim.adam.Adam) performs a single optimization step optimizer.step()

(15) batch stats is added to running_stats (end for3) computes a np.mean for each stat and this mean becomes an epoch_stat. Each epoch_stats is appended into stats. (end for2) An epoch ends here. batch stats include these keys: loss, avg_embedding_norm, num_non_zero_triplets, num_triplets, mean_pos_pair_dist, mean_neg_pair_dist, max_pos_pair_dist, max_neg_pair_dist, min_pos_pair_dist, min_neg_pair_dist.

(16) scheduler.step() is called. scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,eta_min=params.min_lr)
elif params.scheduler == 'MultiStepLR':scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)

scheduler provides several methods to adjust the learning rate based on the number of epochs

(17) writes and prints stats (end for1)

(18) save final model weights









---

## models/minkloc.py

It only defines a class called MinkLoc.

**class MinkLoc()**: is a child class of torch.nn.Module. The class first calls an _init__() of the super class.

self.model is a new attribute. It is the name of the model in params. It determines what the self.pooling will be. 

it inherits the train() and eval() functions from torch.nn.Module.

eval() sets module in evaluation mode. Equivalent with self.train(False). It returns a Module. train() sets the module in training mode. Also returns a Module.

forward() method is called in _call__ function of nn.Modules. so when we run model(input) the forward method is called.

---

## datasets/dataset_utils.py

**make_dataloaders()**: returns a dict() such as {'train': <torch.utils.data.dataloader.DataLoader object at 0x7f043ecf5460>}.

In it, the original dataset is in a variable **datasets**, which is returned by **make_datasets()** function in the same file. 

**make_datasets()**: Create training and validation datasets.

returns something like {'train': <datasets.oxford.IntensityDataset object at 0x7f77b2abaca0>} which comes from **IntensityDataset()** and **OxfordDataset()** objects in datasets/oxford.py. IntensityDataset() object takes OxfordDataset() object as super class. 

---

## datasets/oxford.py

**class OxfordDataset(Dataset)**: Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project, takes torch.utils.data.Dataset as a superclass. 

If query files are alreaady processed and cached, then load preprocessed query files from the cached file path. Only if it is not there will it load and convert to biearray and such.

the **preprocess_queries()** function read through files such as `weeks/output_week1/pointclouds_with_locations_5m/1520479832089447.bin`, from week1 to week 52. each query is a file path. self.query is something like {key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}, key:...}

positives and negatives here refer to positive and negative point clouds. 

**make_collate_fn()**: returns a **collate_fn(data_list)** function according to its inputs. **train_collate_fn** and **val_collate_fn** are both gotten from make_collate_fn(). 

Here the `batch` dictionary returned in the inner function contains the batched spherical coordinates and features. 

**to_spherical()**: turn coordinates to spherical coordinates