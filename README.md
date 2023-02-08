<<<<<<< HEAD
# Write your model faster with pytorch-lightning-wadb-code-backbone
This repository provides the base code for pytorch-lightning and weight and biases simultaneous integration + hydra (to keep configs clean).
This repository shows a **Toy configuration of CV classificator**

**pytorch-lightning-wadb-code-backbone organization** 

```

│   README.md
│   method.py
|   dataloader.py
|   train.py
|
└───models
│   │   model.py
|
└───datasets
│   │   dataset.py
│   │   transformfactory.py
|
└───configs
│   │   defaults.yaml
│   └─── dataloader
│   │    │  dataset.yaml
│   │
│   └─── model
│   │    │  model.yaml
```

### Code structure ###
The code is divided into a number of subpackages:
- models
- datasets
- configs

### How do I use this code ###
The core of this repository is that the pytorch-lightning (pl) pipline is configured though .yaml file.
There are few key points of this repository:
- write your data preprocessing pipline in dataset file (see the **toy** dataset.py and transformfactory.py)
- write your model and pl logic in model file (see the **toy** model.py)
- configure your pipline through .yaml file and see all metrics in [wadb](https://docs.wandb.ai/)

## Quickstart

Login to your `wandb` account, running once `wandb login`.
Configure the logging in `conf/logging/*`.


---


Read more in the [docs](https://docs.wandb.ai/). Particularly useful the [`log` method](https://docs.wandb.ai/library/log), accessible from inside a PyTorch Lightning module with `self.logger.experiment.log`.

> W&B is our logger of choice, but that is a purely subjective decision. Since we are using Lightning, you can replace
`wandb` with the logger you prefer (you can even build your own).
 More about Lightning loggers [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html).
### Configs ###
To understand the structure see [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/).
dataset.yaml and model.yaml consist of dataset_type and model_type keys respectively. Through keys values pl pipline is configured.

**Use case:**
Write your dataset pipline (includes preprocessing through transformfactory.py). Pass dataset_type name (as a key) dataset class (as a value) into self.dataset_types dict in dataloader.py file. 

Write your model pipline (includes with train step logic, see **toy** example). Pass model_type name (as a key) model class (as a value) into self.model_types dict in method.py file. 

Done. 

Configure all parameters through .yaml file with integrated [wandb](https://docs.wandb.ai/)
=======
# graph_edge_generation
graph_edge_generation
>>>>>>> 02870f148f8abf7c90f23a1b1e153a9f9a942101
