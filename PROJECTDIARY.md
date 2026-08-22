Project Diary/ Work Timeline

Term 1

Week 2: I had a meeting with my project supervisor on October 1st and discussed the ideas I had regarding how I want to proceed with my project and how I want to link the topic with its application.

Week 2-3: After the first meeting I started working on my project plan which was due on October 10th. I created a full timeline, researched my ideas to convert them into a proper plan. I found the relevant sources which would be required for research. I assessed the viability of the project and finally decided on "Regression Algorithms for Learning & Its Application on Crypto Currency Market" as the topic for my project.

Week 3-4: I proceeded upon the project plan that was developed and went through the first two books that needed to be read on the topic "The mathematical foundations of regression algorithms, focusing on linear, ridge, and lasso regression" in order to write the first report.

Week 5-6: I wrote the first report covering topics "An overview of regression algorithms, explaining the training/test set concept, cost functions, and derivation of the ridge regression formula." This week I encountered my first difficulty where the plan involved writing a proof of concept program however due to the workload from other modules I was unable to do that part so that is now part of the plan for week 7. I also had my second meeting with my supervisor where he showed concern about nothing showing up on gitlab and I told how I was confused about updating my work on git if my focus for those weeks was only on reading or writing a report and not coding. He told me I should at least update the Diary on git 2-3 times a week which I did not know of since I missed the first fyp lecture.
 
Week 7: I am currently working on the proof of concept program so I can be back on track based on the project plan timeline. 

End of Week 7 Update: I have completed the proof of concept program and have pushed it to the git repository. I am currently going through the literature review on "financial forecasting using regression models" which is part of the plan for week 7 and 8.

Week 8: Update 1 Over the past 3 days I have conducted the literature review on the previous regression based applications on the topic of financial forecasting. I have written the second report and have now uploaded it to the git repository.

Week 9: Update 1, Over the past week I have worked on the third report connecting the theoratical information from the first two reports with the proof of concept program thus completing the foundations for my project. I have just completed the report and added it to the git repository.

Week 10-11 : Over the past two week I have worked on the program which takes the proof of concept program and implements it onto the real world btc-usd dataset. I have successfully implemented it onto the dataset and with it have completed the ridge and ols implementation. I have included the sklearn ridge implementation just to compare my implementation with the built in sklearn implementation. I have spent the rest of the week preparing for my interim presentation.

Week 12: After the presentation i have made one final change to the code by removing sklearn ridge implementation and replacing it with lasso regression. I have committed and pushed the code onto my git repository. I have made a slight change in the alpha value used as now the model checks which alpha value delivers the best result. With this all of OLS, Ridge and Lasso regression have been implemented. Now I am finishing up my interim report and other things needed for the interim submission.

Term 2

Week 1-3: The first 3 weeks of Term 2 were spent on converting the code for the prediction algorithm into an Object Oriented Design instead of one single procedural implementation. I divided the code into seperate classes with each class having a seperate file. This process took a while since the code had to be altered a bit to fit the new design and I had to make sure there were no errors and once that was done everything was synced with the Git repository. During this time I had my first meeting of the term with my supervisor where we discussed my performance for term 1 and my plans for term 2. 

Week 4: Week 4 was spent on working to add K nearest neighbour algorithm to the prediction model. The logic has been added and just a few minor tweaks needs to be fixed.

Week 5: The minor tweaks were fixed in the code and now the model has 4 different algorithms working OLS, Ridge and Lasso Regression which were completed in term 1 and K Nearest Neighbour which has just been added. Everything was then synced with the Git repository. Finally I have started working on adding neural networks to the model and this will be continued in the next week.

Week 6: A small feed forward neural network has been added to the model. The model has been tested to make sure there are no bugs while running it. The parameters of the neural network has been fixed to a certain range bu manually testing to make sure that the model works and the run time is not out of control.

Week 7-8: A Graphical User Interface has been added to the model which allows the user to be able to see the model results in a better way. This interface allows the user to run the code to see the price predictions of the 5 algorithms then the user can also see the historical performance comparison between the 5 algorithms. Finally they can go to the specific tab of each algorithm which has the title, one line description and the two graphs showing the expected vs actual of both the return and price.

Week 9: A major change has been made to the model which now allows the user to choose from 25 different handpicked crypto currency pairs instead of the  fixed BTC-USD pair and also allows the user to choose the number of years for which they wish to use the model between 3-10 years. Also the user can now see the price in both USD and GBP.

Week 10: Kernal Ridge Regression Algorithm has also been added to the model. This is now the 6th and final algorithm. While going through the project description this was one lacking that was discovered and so this model has been added. Also since, the project is now being run through the GUI files the initial main file has still been kept but edited so that the code can be run through the console implementation however in that case only BTC/USD pair for 10 year data can be run while in the GUI implementation all 25 pairs for the data and currency choice of the user can be run.

Week 11-12 + Extension: ReadMe.md file and other supporting files have been added with the correct instructions to run the code. The final report for the project has been written and the professional issues report of the project has also been written. Final minor bugs in the program have also been fixed which allow the program to be more bug proof regardless of the version of imports used.