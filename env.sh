set -e

#sudo apt-get update
#sudo apt-get install -y openjdk-11-jdk

# Create env
conda create -n red python=3.9
source activate red
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

python -m pre_processing.build_contents_index