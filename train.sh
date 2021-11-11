# case 1 : training with params using Fashion MNIST dataset
python train.py \
	--dataset fmnist
	--num_classes 10 \
	--input_shape 28 28 1 \
	--lr 0.01 \
	--momentum 0.9 \
	--epochs 10 \
	--batch_size 32 \
	--verbose 1 \
	--n_folds 5 \
	--saved_model_location models/base_model.h5


# case 2 : training model on generic dataset using image and label files
python train.py \
	--dataset fmnist
	--num_classes 10 \
	--input_shape 28 28 1 \
	--lr 0.01 \
	--momentum 0.9 \
	--epochs 10 \
	--batch_size 32 \
	--verbose 1 \
	--n_folds 5 \
	--saved_model_location models/base_model.h5

