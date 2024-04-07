
## A Brief of the Prototype:

Quality education is the path to transcendence and making this journey more engaging is the sole purpose of our project. We intend to include features that would provide an interactive interface to the user where he/she can have a more precise watch on his/her performance, explore their areas of interest, and connect with people of the desired field in the vicinity using the real-time trained DL and ML model. 

Making the learning more innovative we would allow them to create chatrooms for discussions on projects, academics, areas of interest, etc., and also help them with auto-generated suggestions. 

Creating fun and amazing pet avatars in which the capital to unlock new features would be through their scores that would be auto-generated based on their day-to-day progress in activities as well as academics.

Apart from learning, we would also take care of their mental health by creating interactive chatbots which could judge a child's mental state and provide suitable help. Thus, providing a pool of exploration alongside fun and healthy competition would help in the overall development of the student.

## How it works!

The user interactions related to learning would be duly recorded. These recordings will be taken as input features for our Intel oneAPI oneDNN and oneDAL integrated ML and DL models. These models will further make predictions, cluster the students according to their areas of interest, generate visualization for their progress in each of their domains/subjects, and provide the candidates with study material recommendations whether it is a video, blog, or book.
Some fun activities and assessments to help them, co-op with their studies, which in turn will be recorded and a ranking system amongst the students say class-wise, section-wise, or campus-wise will provide them scores for performing well in those activities and classes and rank them accordingly, creating a healthy competitive environment.

During Covid-19, when everything had shifted to online mode, students used to get bored and drowsy during classes and it was not possible to keep a check on each and every student. Through our project, we would train the model to check if the person is getting drowsy and recommend him some auto-generated exercises that would help him get his focus back. Most institutions stick to the old school analysis systems in which the student is never able to understand specifically in which area, he/she should improve in order to maintain his academic results, but through our project, he/she can get a detailed analysis of his performance with day to day updates which will keep him/her more aware about his situation in every area.

# List of features offered by the solution

## Vision

Making learning more interactive and innovative by providing a clearer picture along with some fun elements, here we list the features of our project bifurcated into three categories.

## Category - 01   (Academic)
01. An advanced ranking system that will be updated on a day-to-day basis which would be based on the overall performance during lectures, projects, assignments, and quizzes.

02. Keeping track of the attendance of the student in every subject and sending reminders to attend lectures on subjects in which his attendance is going down at an alarming rate.

03. Providing him/her proper insights about the areas he is currently lacking in along with auto-generated suggestions of resources to study from. 

04. A separate area for assignment submission along with a plagiarism checker. The plagiarism checker would keep on updating itself with each assignment submission. If it receives an assignment with exactly the same content, the student would receive a notification to resubmit another assignment as the presently submitted assignment might be up for plagiarism.

05.  A highlight section that would keep them connected to the world. Giving them a daily insight into what's happening around the globe, especially their areas of interest.

## Category - 02  (Gamification)
01. Improvisation of the ranking system through day-to-day digital badge system. Every day the student who has outperformed everyone else by giving more quizzes and doing his assignments would be provided with the student of the day batch.

02. Interactive quizzes through facial gestures. Multiple choice  questions would be answered with the movement of the head. 

03. Adding some fun virtual elements, there will be a performance-wise token credit system in this Gamification Centre. Based on this credit, the students can unlock new features for their virtual pet. 

## Category - 03  (Student Insights)
01. Checking on the mental health of the student, there will be interactive mental health chatbots. In case the student needs any kind of help, he/she can be provided immediate assistance.

02. Detection of drowsy and low behaviour during online lectures. If the student has been feeling drowsy for a certain period of time during the lecture, then some easy sitting exercises will be suggested in order to tackle the drowsiness. 

03. The students can create chatrooms where all the interested can pitch in for a project, assignment, etc. In case they need a mentor, then we can provide them with suggestions from the faculty experts in that field.
# Student Login/Register

Firstly we present you the login and register page. First time visitors will have to register themselves to the portal in order to enjoy the benifits of the dashboard. They also need to enter their favourite category which would influence the outputs generated by the recommendation system. The already registered students just need to fill in the email and password.

# Dashboard

Now the new student is welcomed with the dashboard welcome message. Here we have the weather API that would Fetch the current information about the weather and keep the dashboard more updated with the outside world, keeping the student well informed. Nextly we have the rank of the student represented in the form of graphs. These graphs display the rank class-wise, section-wise and school-wise separetely. So that students are always well informed of their performance and the scope of improvement.
Now there will be a alert or notification section as well, where the alerts from the admins will be presented. The next section includes the attendance for all the subjects along with the average attendance. Next is a piechart showing the number of assignments submitted where the student will get to know how many assignments are still left to submit. After that, we have the Learners'Ed Coin Bank, which tells us how many coins are left with the user and on the right side we have our virtual pet information, such as pet name, level and rank.
At last, we have the recommendation section where the category selected would determine which type of videos will be fetched for the student and be presented in the recommendation section. 

# Lecture Section

Now we have the lecture section where all of the lectures recommended for the students are listed here. The student can watch that lecture by clicking on the watch button after going into the lecture section. We are also introducing a 'Drowsiness Detection System', in which if any teacher uploads a lecture, this system would track down the time for which a student has been actively listening during the lecture and would provide the attendance accordingly. However, we understand, that online lectures can get a little boring and tiresome. So, in order to tackle that when the system would notice that the student hasn't been attentive for a certain period of time, it would pause the video and generate a promt that would suggest an easy sitting exercise that could help them regain their focus and not miss on the learning. Once he/she has done the exercise and is fresh again, he/she can click on the 'OK' button and continue with his lecture.

# Drowsiness Detection System
 
You can acces the drowsiness detection model used behind this, present in the AIML Modules Folder.

The provided code in the AIML modules folder implements a drowsiness detection model using Convolutional Neural Networks (CNNs). Here is a summary of the code functionality:

Importing necessary libraries: The code starts by importing the required libraries, including PIL, OpenCV, face_recognition, TensorFlow, and Keras.

Eye Cropping Function: The eye_cropper function takes an image path as input and uses OpenCV and face_recognition to locate the eyes in the image. It crops the eye region, resizes it to 80x80 pixels, and returns the cropped image for further processing.

Loading Images from Dataset: The load_images_from_folder function loads images from the specified folder and resizes them to 80x80 pixels. It assigns a label (0 for open eyes, 1 for closed eyes) and creates a list of image-label pairs.

Preparing the Dataset: The code creates arrays for input images (X) and corresponding labels (y). It iterates through the image-label pairs, appends the images to X, and labels to y. The images are reshaped, normalized, and the labels are converted to arrays.

Splitting the Dataset: The dataset is split into training and testing sets using the train_test_split function from sklearn. The splitting is stratified based on the labels to maintain class balance in both sets.

Model Definition: The code defines the CNN model using the Sequential API of Keras. It includes convolutional layers, max-pooling layers, dense layers, and dropout layers for regularization.

Model Compilation and Training: The model is compiled with binary cross-entropy loss and Adam optimizer. It is then trained on the training data, using the fit function, for 24 epochs. The validation data is used to monitor the model's performance during training.

Model Evaluation: The trained model is evaluated on the testing data using the evaluate function. The evaluation results, including loss and metrics, are printed.

Model Saving: The trained model is saved to a file using the save function.

Prediction Function: The model_response function takes an image and uses the eye_cropper function to extract the eye region. The preprocessed image is then passed to the trained model for prediction. If the predicted probability of closed eyes exceeds a threshold, the function returns 'Yes,' indicating drowsiness.

Model Usage: The model_response function is called with an image to demonstrate the usage of the trained model.

In summary, the code prepares and trains a CNN model to classify eye states as open or closed for drowsiness detection. It provides a function to extract eye regions from images and a function to classify the eye state using the trained model.
## Chatroom

The first one is chatrooms, this chatrooms are available throughout the campus, further these can be sub divided into class-wise, section-wise as well, so that students can chat and get along their coomunity. This will help building a community mindset amongst them and never leave them alone in the lockdown period of covid, making them 
mentally strong. 


## Mental Health Chatbot

The second section includes mental health chatbot which is a deep learning model trainged chatbot. This model helps the students to discuss their problems with them and give a stress release session with the chatbot. The chatbot gives a helping hand to the students if they face any such mental health issues.


# Summary of Intel oneAPI AI Analytics Toolkit Optimization
To meet the project requirements, we have developed three sophisticated Deep Learning Models that play crucial roles in different sections of our platform. These models have been enhanced with Intel oneDNN and OpenMP optimization, allowing us to achieve exceptional performance gains, including faster training, higher throughput, improved inference speed, and reduced latency.

The first model we have developed is the Drowsiness Detection Model, which is utilized in the Lecture Section. By leveraging Intel oneDNN and OpenMP optimization, we have significantly accelerated the training process of this model. This optimization framework has not only expedited the training phase but has also improved the overall inference speed during real-time drowsiness detection. As a result, students' attendance can be accurately determined based on their active listening time during lectures.

The second model we have incorporated is the Face Pose Estimation Model, which plays a crucial role in the Gamify Quiz section. Through the implementation of Intel oneDNN and OpenMP optimization techniques, we have achieved remarkable improvements in training efficiency, throughput, and inference speed. These optimizations have enabled us to accurately estimate facial movements and gestures during the quiz, providing an engaging and interactive learning experience for students.

Lastly, we have developed a sophisticated Mental Health Chatbot that leverages Intel oneDNN and OpenMP optimization. This optimization framework has significantly enhanced the performance of our chatbot, resulting in faster response times, improved throughput, and reduced latency. By employing Intel's optimization tools, such as scikit-learn-intelex, we have been able to train the chatbot model efficiently and deliver prompt and insightful responses to students seeking support.


<h1 align="center">Thank You</h1>


