# Créer un environement CONDA et installer les requiremetns
conda create --name $ENVIRONMENT_NAME python tensorflow  --file requirements.txt

conda create --name cnn_raphael python=3.9  --file requirements.txt
et ou 
pip install -r requirements.txt

# pour activer l'environement CONDA
conda activate cnn_raphael

# pour le deactiver
conda deactivate 

## Pour créer un docker 
docker build -t cnn-docker .

# Pour lancer un docker
docker run cnn-docker

# dashboard 

%load_ext tensorboard
%tensorboard --logdir logs