# case 1 : Evaluating fmnist test dataset
python predict.py \
	--input_shape 28 28 1 \
	--verbose 1 \
	--saved_model_location models/base_model.h5


# case 2 : Predicting a new test image
python predict.py \
	--input_shape 28 28 1 \
	--verbose 1 \
	--saved_model_location models/base_model.h5 \
    --use_image \
    --image_location images/test1.png \
	--custom_labels 't_shirt_top' 'trouser' 'pullover' 'dress' 'coat' 'sandal' \
			'shirt' 'sneaker' 'bag' 'ankle_boots'