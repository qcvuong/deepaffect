================
================
DEEP AFFECT DEMO
================
================

November 10, 2021

Please cite: "Parry, G., & Vuong, Q.C. (2021, June 8). Deep Affect: Using objects, scenes and facial expressions in a deep neural network to predict arousal and valence values of images. https://doi.org/10.31234/osf.io/t9p3f"



===================
Training/validation
===================

_train_FinalProccessingKFoldREG.py	: Main training python script



_train_TRAINdata.csv			: CSV file containing training data
					  Must contain at least header: FileName, AroMean (arousal mean) & ValMean (valence mean)
					  For filename, don't include extension


_train_VALIDATIONdata.csv		: CSV file containing validation dat
					  Must contain at least header: FileName, AroMean (arousal mean) & ValMean (valence mean)
					  For filename, don't include extension


<__builds__>
	* Contains necessary scripts for CNNs, etc.
	* Examples of DEEPAFFECT model builds are:
	FullnetMark2.py
	FullnetMark3.py
	FullnetMark3_Freeze.py * currently set to use this model build


<full_imgs_train>
	* need to create this folder (with this filename) in the <source> folder
	* Put all training/validation images here (currently set for .jpg images)



COMMAND LINE:
python _train_FinalProccessingKFoldREG.py --model myModel.h5 --structure myModelStructure.png --Logger myModelResults.csv --EarlyStop myModel_earlystop.h5 --plot myModelPlot.png


OUTPUTS in the main folder <source>:

myModel.h5				: trained model


myModel_earlystop.h5			: temporary trained model at each epoch, can delete at the end of training (large file)


myModelStructure.png			: model structure


myModelResults.csv			: training/validation results (loss over epochs, same as plot)


myModelPlot.png				: plot of training/validation loss over epochs




===================
Test
===================

_test_Prediction.py			: Main test python script


_test_data.csv				: CSV containing test data
					  Must contain at least header: FileName, AroMean (arousal mean) & ValMean (valence mean)
					  For filename, don't include extension


<full_imgs_test>
	* need to create this folder (with this filename) in the <source> folder
	* Put all test images here (currently set for .jpg images)


<aaPredictions>
	* need to create this folder (with this filename) in the <source> folder
	* Output of predictions stored in this folder
	PREDICTIONS.csv: predicted valence and arousal (separated by ----) the first set of predictions (rows) are for valence, then separator (----), then arousal
	AROUSAL.csv: human rated arousal score (copied from _test_data.csv)
	VALENCE.csv: human rated valence score (copied from _test_data.csv)
	LABELS.csv: image filename (without extension) (copied from _test_data.csv)


After training, the trained model (myModel.h5) can be used to predict arousal/valence of new images
OR else use any desired previously trained model (a.h5 file)


COMMAND LINE:
python _test_Prediction.py --model myModel.h5

* make sure that the model is in the <source> folder with this script


OUTPUTS:

See files stored in <aaPredictions>




=========
=========

12/11/21, qcv
