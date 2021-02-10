import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TorchDiffEqPack", 
    version="0.1.0",
    author="Juntang Zhuang",
    author_email="j.zhuang@yale.edu",
    description="PyTorch implementation of reverse-accurate ODE solvers for Neural ODEs",
    long_description="PyTorch implementation of two papers: (1) Adaptive checkpoint adjoint method for gradient estimation in Neural ODEs (2) MALI: a memory efficient and reverse accurate integrator for Neural ODEs",
    long_description_content_type="text/markdown",
    url="https://juntang-zhuang.github.io/torch_diff_eq_pack/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'torch<1.7.0',
      ],
    python_requires='>=3.6',
)
