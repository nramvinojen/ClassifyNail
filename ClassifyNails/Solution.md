1. How well does your model perform? How does it compare to the simplest baseline model you can think of?
  As the results show the base line model ahd an average tes accuracy around 67%
  While the test accuracy of the custom vgg16 model was around 95%
	can be improved with giving more attention to the hyperparamete, 
	bayes hyperparameter tuning can be implemented to further improve
  
2. How many images are required to build an accurate model?
  At least few thousands of images would be needed to build an accurate model, 5000 images would do. 
  Augmentation of the data like horizontal/vertical, flipping, rotation, zoom will definetely be useful.
  
3. Where do you see the main challenge in building a model like the one we asked here?
chosing the perfect hyperparametes like optimization algorithm, learning rate etc.
Also acounting for overfitting.

4What would you do, if there would be only 20 bad images and 100 good images?
oversampling one class, and undersampling the other class will help.
Also interpreting the result metrics, such as precision, recall and f1 score


5. What problems might occur if this solution would be deployed to a factory that requires automatic nails quality assurance?
Prediction speed
	the model has to as light as possible 
	also the SW infrastructure for the deployments plays a vital role 
Lighting condition 
	personally i belive the background and lighting condition can be made better than the give dataset
	the background of the provided dataset is not the ideal one
Accounting for non ideal cases such as nails grouping together
