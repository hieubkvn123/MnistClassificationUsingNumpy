# MnistClassificationUsingNumpy

This code snippet will show you how to create a multi-layer neural network (including 
back-probagation) to train a mnist hand written digits classifier using nothing but numpy.
In order to train the model, run the pre-made shell script run.sh :
```bash 
$ ./run.sh 
```

If you want to run the training script with your own configuration, run the python file
with the config variables of your choice.
```bash
$ python3 train.py --epochs <EPOCHS> \ # number of train iterations
				   --batch_size <BATCH_SIZE> \ # number of images per batch
				   --lr <LEARNING_RATE> \ # The learning rate
				   --checkpoint_step <STEPS> \ # How often you want the model to checkpoint
				   --model_path <PATH> # The model's weights file
```

# Training results :
![None](images/results.gif?raw=true "Training results")
