# Installation instructions
**1. Create a conda virtual environment and activate it.**
```shell
conda create -n cotr -y python=3.8 conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja
conda activate cotr
```

**2. Install Pytorch following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3
```


**3. Install mmcv-full.**
```shell
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

**4. Install mmdet & mmsegmentation.**
```shell
pip install mmdet==2.25.1 mmsegmentation==0.25.0
```

**5. Install other requirements.**
```shell
pip install lyft_dataset_sdk networkx==2.2 numba==0.53.0 numpy nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39
```

**5. Prepare cotr repo by.**
```shell
git clone https://github.com/NotACracker/COTR.git
cd COTR
pip install -v -e.
```

