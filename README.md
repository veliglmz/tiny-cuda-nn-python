# Python Implementation of tiny-cuda-nn

This code is an unofficial Python implementation of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp). I've implemented only the [multiresolution hash encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) paper.

[Numba](https://numba.pydata.org/) library is used to accelerate.

## Requirements
* Docker 24.0.5
* Docker Compose v2.20.2

## Usage

Clone the repository.

```bash
git clone https://github.com/veliglmz/tiny-cuda-nn-python.git
cd tiny-cuda-nn-python
```

Build the docker image.
```bash
docker compose build
```

Run the docker image. (the outputs are in the results folder of the host.)
```bash
docker compose run app
```

Stop containers and remove containers, networks, volumes, and images.
```bash
docker compose down
```

