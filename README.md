# UNSUPERVISED TEMPORAL CONSISTENCY IMPROVEMENT FOR VIDEOSEGMENTATION WITH SIAMESE NETWORKS

This repository provides code for "UNSUPERVISED TEMPORAL CONSISTENCY IMPROVEMENT FOR VIDEOSEGMENTATION WITH SIAMESE NETWORKS"

## Code Structure
Paths to data should be provided in *data_config/datasets_config.py* (there are some ready examples).
In the config file one also should provide information about focus plane indexes. Precomputed (or manually labeled) information about indexes is contained in *focused_frames/* directory

*utils* contains some additional training/visualization/logging helper functions.

As described  in the paper, model training consist of 2 parts: 
1) training a model on segmentation task (*train_model.py*), 
2) training the same model on segmentation **and** temporal consistency tasks (*train_seq_model.py*). 

Scripts for test: 
* *run_test_predictions.py* - script for saving predictions for all data specified in *datasets_config.py*
* *eval_model.py* - model evaluation

## How to run

### Training
First training step:
```
python train_model.py -name <model name> -NUM_CHAN <num channels to use in z-stack, default 7> -cuda <cuda id> -DATA_TYPE <NUCL for nucleoli, TRITC for nuclei>
```

Second training step:
```
python tra_seq_model.py -BASE_MODEL_NAME <simple model name> -TIME_LEN <temporal learning window> -lr_seg <segmentation learning rate> -lr_time <temporal consistency learning rate> -ADD_NAME <additional log comment> -TIME_LOSS <temporal consistency loss type> -DATA_TYPE <NUCL for nucleoli, TRITC for nuclei> -cuda <cuda id>
```

### Evaluation
Saving predictions:
```
python run_test_predictions.py -name <model name> -TTA -DATA_TYPE <NUCL for nucleoli, TRITC for nuclei> -cuda <cuda id>
```
Evaluation script:
```
python eval_model.py  -name <model name> -DATA_TYPE <NUCL for nucleoli, TRITC for nuclei>
``` 