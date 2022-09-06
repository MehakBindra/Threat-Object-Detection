# Threat-Object-Detection

Put the input knife images- "images/knife/" directory
Put the input scissors images- "images/scissors/" directory

Now run preprocess.ipynb ob jupyter notebook

For training the model- 

Using lenet- "python train_lenet.py --dataset images --model lenet.model -p graph1"

Using resnet- "python train_resnet.py --dataset images --model resnet.model -p graph2"

Using vgg- "python train_vgg.py --dataset images --model vgg.model -p graph3"

For testing -

On lenet.model- "python test_network.py --model lenet.model --image images/examples/name_of_img.jpg/png"

On resnet.model- "python test_network.py --model resnet.model --image images/examples/name_of_img.jpg/png"

On vgg.model- "python test_network.py --model vgg.model --image images/examples/name_of_img.jpg/png"




