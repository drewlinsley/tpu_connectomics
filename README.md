## Connectomics

# setup environment
create a new machine: 
```bash
gcloud compute instances create test-tpu --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform
gcloud compute ssh test-tpu --command "sudo apt-get install python-opencv libsm6"
```

install python libraries
```bash
pip3 install -r requirements.txt
```

## data
The data can be found for multiple different dimensions in `gs://serrelab-public/data/`.

## run 32x32x24 version
First, edit the `run` file with the configurations that work for you.
Each of the dimensions can be changed in the configuration file `configs/default3d.conf`
(x, y, z) = (input_shape[0], input_shape[1], z_slices)
```bash
./run
```
