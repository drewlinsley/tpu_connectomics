## Connectomics

# setup environment
create a new machine: 
```bash
gcloud compute instances create test-tpu --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform
gcloud compute ssh test-tpu "sudo apt-get install python-opencv libsm6"
```

install python libraries
```bash
pip3 install -r requirements.txt
```

# run 32x32x24 version
```bash
. ./load_env
python3.5 train.py
```
