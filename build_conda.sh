conda create -n confMILE python=3.8
module load cuda/11.8
conda install tensorflow
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install gensim pandas theano scikit-learn
conda install -c dglteam/label/cu118 dgl
conda install -c conda-forge pytorch-lightning