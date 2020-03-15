# Nail classification

Clearly the data set is small hence using transfer learning techinque is benificial.
I use a Convolutional neural network model to classify industrial nail images as good or bad.

I use VGG16 pre trained network with custom layers at the top as for the classification task. 
	I choose VGG16 beacasue I have worked on similar task like this before
	I prefer VGG16 over many other (Resnet50), due to ints high accuracy and light weight

A base line classification accuracy is establised wtih a simple CNN model. 


A simple Data augmentation, cropping pipeline is setup for the model.
The hyperparameter tuning is done manually and from emprical results.   


### Prerequisites 

The requirements.txt file contains all the necessary packages for the whole task.
I use python 3.6
Run the following file to install all the required packages
```
python install_req.py
```

### Steps for Basic Running 

Train a model from scratch:
- Copy the nail images to /data folder in the format /data/whole.
```
    └── data
        └── whole
        │   ├── good
        │   └── bad
        └── split
	    ├── train
	    ├── validate
	    └── test
	
  ```
			

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
  * This will train the model and show training and validation loss and accuracy values of both the baseline model and vgg model.
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
-For classifying an image using vgg16 model:
 using Curl
 ``` 
 curl --location --request POST "http://localhost:5000/predict"  --form  image=@C:/1522072665_good.jpeg
 ```
 using Postman
 Open send the image in "form-data"
   
```   
 set "key" as image and add the image in "value" under the "body" tab
```





