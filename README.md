# Highway Networks (Training Very Deep Networks)
This is my replication of the Highway Networks Model using the Cifar10 Dataset

## Setup and Running
This project was created using `uv` and is highly recommended<br>
After installing `uv` this project should run out of the box<br>

### Data Setup
You can get the original Dataset from [Alex Krizhevsky's webpage](https://www.cs.toronto.edu/~kriz/cifar.html) although there are likely sources out there that provide it in a more modern format. 

Once this format is downloaded you will need to run the `cifar10_data_processing.py` script to create our datasets<br>
```
uv run cifar10_data_processing.py
```
This will create a data folder with both training and test directories

### Training
Before kicking off training you should update the weights and biases variables `entity` and `project` in `init_logging()` in train.py to match your account.<br>
If not using Weights and Biases (not recommended) you can set logs to `False` in main.py<br>
To kick off training you can run<br>
```uv run main.py```

# Citation 
The initial paper titled ["Highway Networks"](https://arxiv.org/abs/1505.04597) <br>
The full-length Paper entitled ["Training Very Deep Networks"](https://arxiv.org/abs/1507.06228)

```bibtex
@article{DBLP:journals/corr/SrivastavaGS15,
  author       = {Rupesh Kumar Srivastava and
                  Klaus Greff and
                  J{\"{u}}rgen Schmidhuber},
  title        = {Highway Networks},
  journal      = {CoRR},
  volume       = {abs/1505.00387},
  year         = {2015},
  url          = {http://arxiv.org/abs/1505.00387},
  eprinttype    = {arXiv},
  eprint       = {1505.00387},
  timestamp    = {Mon, 13 Aug 2018 16:48:21 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SrivastavaGS15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/SrivastavaGS15a,
  author       = {Rupesh Kumar Srivastava and
                  Klaus Greff and
                  J{\"{u}}rgen Schmidhuber},
  title        = {Training Very Deep Networks},
  journal      = {CoRR},
  volume       = {abs/1507.06228},
  year         = {2015},
  url          = {http://arxiv.org/abs/1507.06228},
  eprinttype    = {arXiv},
  eprint       = {1507.06228},
  timestamp    = {Mon, 13 Aug 2018 16:48:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SrivastavaGS15a.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```