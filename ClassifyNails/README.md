# Nail classification


Convolutional neural network based model to classify manufactured nail images as good or bad (bent).
As the data set is small, transfer learning is a highly recommended method for the task. A pretrained model is used 
as the building block for the classification task. For this task, a pretrained vgg-16 model with a customized top 
is implemented using keras.
A simple CNN model is also implemented to establish a baseline. Data augmentation and hyperparameter tuning helped 
achieve plausible results.   


### Prerequisites 

The requirements.txt file contains all the necessary packages for the whole task.



### Steps for Basic Running 

Train a model from scratch:
- Copy the nail images to /data folder in the format /data/whole.
 

For this task first copy the images into the data folder.  The tree structure starting from the root directory should look like 

which creates 

    └── data
        └── whole
        │   ├── good
        │   └── bad
        └── split
  
			

and then execute

```
python easy_run.py mkds
```
which creates 

    └── data
        └── whole
        │   ├── good
        │   └── bad
        └── split
            ├── train
            |   ├── good
            |   └── bad
            ├── validate
            |   ├── good
            |   └── bad
            └── test
                ├── good
                └── bad
				
- Run the model:
  * This will train the model and show training and validation loss and accuracy values.
```
python easy_run.py train
```

- test and conmpare models:
  * This will load the trained models and check the results on unseen test data.
```
python easy_run.py test
```

-results
models tested and compared  in test data set
```
---------------Results of the  cnn model ---------------

-------------Confusion Matrix
[[7 5]
 [3 9]]

-------------Classification Report
              precision    recall  f1-score   support

        good       0.70      0.58      0.64        12
         bad       0.64      0.75      0.69        12

    accuracy                           0.67        24
   macro avg       0.67      0.67      0.66        24
weighted avg       0.67      0.67      0.66        24

Found 24 images belonging to 2 classes.


---------------Results of the  vgg16 model ---------------

-------------Confusion Matrix
[[10  2]
 [ 0 12]]

-------------Classification Report
              precision    recall  f1-score   support

        good       1.00      0.83      0.91        12
         bad       0.86      1.00      0.92        12

    accuracy                           0.92        24
   macro avg       0.93      0.92      0.92        24
weighted avg       0.93      0.92      0.92        24
```



- Run the server:
```
python app.py
```
- For classifying an image using vgg16 model:
   Open "Postman" send the image in "form-data"
   
```   
 set "key" as image and add the image in "value" under the "body" tab
```





