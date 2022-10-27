# Python Implementation of tiny-cuda-nn

This code is an unofficial Python implementation of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). I've implemented only the [multiresolution hashing encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) paper.

To accelerate, [Numba](https://numba.pydata.org/) library is used.

## Requirements
* CUDA 11.3

## Installation

* Clone the repository
* Create a virtual environment
* Install pip packages.
If you have trouble with torch, please install it according to [PyTorch](ttps://pytorch.org/).

```bash
git clone https://github.com/veliglmz/tiny-cuda-nn-python.git
cd tiny-cuda-nn-python
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py -i data/wall.jpg -c config.json -o inferences
```
