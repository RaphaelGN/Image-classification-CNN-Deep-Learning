![Demo](./show_photo/demo_image.png)
## Why Learn CNNs?

CNNs have become fundamental in computer vision and image analysis. They are behind cutting-edge technologies like image recognition, object detection, and more. Learning CNNs can open up exciting career opportunities and enable you to create innovative applications.


# Créer un environement CONDA et installer les requirements
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

Happy learning! 🚀
