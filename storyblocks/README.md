# storyblocks sagemaker deep learning container development

this repo holds all storyblocks custom sagemaker deep learning development files. at a high level, the process is:

1. create a sub-directory here to hold your application
2. create the model files that will be archived and sent to sagemaker
    + e.g. `inference.py`, `model.pth`, etc
    + different per modelling framework
        + e.g. for `pytorch`,
          review [this](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#bring-your-own-model)
    + when in doubt, go to a previous storyblocks model folder (e.g. `object_detection`) and copy
        + the `sh` files
        + `README.md`
        + `model/code/inference.py`
        + `model/code/requirements.txt`
3. build the base container locally using one of the `Dockerfile`s under your chosen framework
   + this will act as a local version of the container you will select when you deploy to sagemaker
   + this can take a *long* time
   + make sure the version number you are building exists as a selectable version number on sagemaker (this repo can get
   + ahead)
4. run the built container from (3). When you do, mount your custom files into `/opt/ml/model`
5. develop locally: verify everything works, write any needed tests, etc
6. write a script to archive your custom model files (whatever ended up in `/opt/ml/model`)
7. push that archive to `s3` and deploy using sagemaker sdk code (see previous implementations for examples)
