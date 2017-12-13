# Deep Learning with Topological Signatures

This repository contains the code to reproduce the results of the following paper:

```bash
@inproceedings{Hofer17c,
  author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
  title     = {Deep Learning with Topological Signatures},
  booktitle = {NIPS},
  year      = 2017}
```

#Read me first:
 1. The intent of this repository is to reproduce the results of *Hofer17c*. If you 
are looking for code optimized for reuse allow me to refer you to [chofer_torchex](https://github.com/c-hofer/chofer_torchex)
and [tda-toolkit](https://github.com/c-hofer/tda-toolkit).
 2. I have tested the code on ubuntu 14.04 and 16.04 system setups. Since this is more or less a two man show 
 testing was not as intensive as it could have been. So if you use the code I consider you as beta tester :). 

# Installation 

1. Ensure ```PyTorch``` is installed properly. (During developement I used ```PyTorch``` 0.2)

1. If you want to calculate the persistence diagrams yourself make sure the tda-toolkit submodule 
is configured properly, see [tda-toolkit](https://github.com/c-hofer/tda-toolkit) for how to do this. 

1. Clone the repo with the ```--recursive``` flag set (otherwise the submodules won't be cloned). If you want to 
install the submodules manually you can omit the flag. 

# Usage 

In order to reproduce the results for a specific data set just run the corresponding scripts in the root folder of the repo, i.e.,
```bash
cd /dir/to/nips2017
python animal.py
```
