# Mobile-Phone-Use-Alert-System-With-Face-and-Time-Stamp-Recording
INTRODUCTION TO THE PROJECT-

This project is aimed to restrict drivers from using mobile phone while driving. This project involves training of custom yolo model to detect a person using mobile phone. On detection, an alert alarm is raised and photo of the person using phone is clicked and saved to a folder along with time stamp of using it. This can be used as a part of driver's distraction detection system.

ABOUT THE DATASET-

A dataset containing pictures of people using phone on left side , using phone on right side and not using phone. The dataset is labelled using open source labelling tool. The labels conatins the coordinates of the detected object.

PREREQUISITE INSTALLATION-

open-cv, numpy, pygame

STEPS OF THE PROJECT-

1. DATA COLLECTION- Pictures of people using phone -left and right and not using phone are collected by clicking images from high resolution camera.

2. DATA LABELLING- The dataset collected is labelled using open source labelling tool to generate a text file containing coordinates of the object to be detected along with their classes. Both the label files and images are transferred to folder named obj and zipped it.

3. GENERATING PATHS OF THE IMAGES FILES- Made a test and train text files containing paths of images to be trained.

4. PREPARATION OF YOLO CFG FILE- Downloaded yolo-v4-tiny.cfg and edited the file by changing number of classes and number of filters (formula used- (classes_num + 5)*3).

5. PREPARATION OF NAMES AND DATA FILE- Made obj.names file containing names of the classes to be detected namely- using_phone_left, using_phone_right and not_using_phone. Made obj.data file in which there are informations like number of classes, paths of backup folder, test.txt, train.txt and obj.names file.

6. PREPARATION OF FINAL FOLDER CONTAINING FILES NEEDDED FOR TRAINING- .cfg file, .names file, .data file, obj.zip, test.txt and train.txt are transferred to a folder and uploaded on drive.

7. TRAINED YOLO CUSTOM MODEL ON GOOGLE COLAB- Trained yolo custom model by cloning https://github.com/AlexeyAB/darknet and using the saved folder in step6. The code of training can be found in training.pynb file.

8. DOWNLOADED WEIGHTS FILE- Downloaded weights file of trained model to further use it for inference.

9. CREATION OF FOLDER- Created a folder to further save the cropped license plates of vehicles.

10. WRITING FINAL CODE FOR RUNNING INFERENCE- Written final code to run inferece on real time pictures using .cfg , .names and .weights file. Alarm file is added. The code is in license_plate1.py. The code when runs clicks pictures of vehicles continuously and send them to trained model to detect the whether person is using phone or not. If it finds the person using phone crops the face of the detected person and saves it to a folder and records the time stamp of using it. The frequency of recording of time stamp and pictures depends on a threshold value which can be set. The alarm sound is raised using pygame.

STEPS TO RUN INFERENCE-

Install the prerequisites mentioned above.
Change the paths of yolov4-tiny-obj.cfg, yolov4-tiny-obj_last.weights, obj.names and output folder in phone_alarm.py. 
Open command prompt at this path and write- python phonr_alarm.py 
The webcam will start and it will start detecting continously unless interrupted using enter key. If found using phone it will save the faces of users and record the time stamp in a text file.  Set the threshold of recording according to your need in .py file.

FUTURE SCOPE OF THE PROJECT-

At present this project can be implemented in driver distraction detecting systems using appropriate hardware. The model is a bit slower on real time if used in mobile devices like raspberry pi. This can be made faster by converting into other faster formats like open vino format(.xml and .bin) or tensorflowlite. But converting this into these forms may lead to less accuracy. So there is always a speed-accuracy trade-off.
