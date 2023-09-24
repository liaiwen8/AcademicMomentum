import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
import tkinter
from tkinter import ttk as tkinter
from ttkthemes import ThemedTk
import seaborn as sns
import matplotlib.pyplot as plt

def enter_data():
    global grade1, grade2, studytime, failures, absences, health, family, freetime

    grade1 = grade1_entry.get()
    grade2 = grade2_entry.get()
    studytime = studytime_entry.get()
    failures = fails_entry.get()
    absences = absence_entry.get()
    health = health_entry.get()
    family = family_entry.get()
    freetime = free_entry.get()



data = pd.read_csv("_internal/student-math.csv", sep=";")
data.head()

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "health", "famrel", "freetime"]]
predict = "G3"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

regression = RandomForestRegressor(n_estimators=100, max_depth=10)
regression.fit(xtrain, ytrain)
accuracy = regression.score(xtest, ytest)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(ytest, regression.predict(xtest))

print("RANDOM FOREST REGRESSION MODEL TRAINED.")
print("ACCURACY: ", end='')
print(accuracy)
print("MEAN SQUARED ERROR (best to be below 4): ", end='')
print(error)



window = ThemedTk(theme='breeze')
window.title("STUDENT GRADE PREDICTION")
frame = tkinter.Frame(window)
frame.pack()

user_info_frame =tkinter.LabelFrame(frame, text="STUDENT INFORMATION")
user_info_frame.grid(row= 0, column=0, padx=20, pady=10)

grade1_label = tkinter.Label(user_info_frame, text="G1 Grade (%)")
grade1_label.grid(row=0, column=0)
grade2_label = tkinter.Label(user_info_frame, text="G2 Grade (%)")
grade2_label.grid(row=0, column=1)
studytime_label = tkinter.Label(user_info_frame, text="Studytime (1 = <2h, 2 = 2-5h, 3 = 5-10h, 4 = >10h)")
studytime_label.grid(row=2, column=0)
fails_label = tkinter.Label(user_info_frame, text="Past Failures of Class (4 if over 3)")
fails_label.grid(row=4, column=0)
health_label = tkinter.Label(user_info_frame, text="Health Status (1-5)")
health_label.grid(row=4, column=1)
family_label = tkinter.Label(user_info_frame, text="Family Relationship (1-5)")
family_label.grid(row=6, column=0)
free_label = tkinter.Label(user_info_frame, text="Freetime After School (1-5)")
free_label.grid(row=6, column=1)
absence_label = tkinter.Label(user_info_frame, text="School Absences (0-93)")
absence_label.grid(row=2, column=1)

grade1_entry = tkinter.Entry(user_info_frame)
grade2_entry = tkinter.Entry(user_info_frame)
studytime_entry = tkinter.Entry(user_info_frame)
fails_entry = tkinter.Entry(user_info_frame)
health_entry = tkinter.Entry(user_info_frame)
family_entry = tkinter.Entry(user_info_frame)
free_entry = tkinter.Entry(user_info_frame)
absence_entry = tkinter.Entry(user_info_frame)

grade1_entry.grid(row=1, column=0)
grade2_entry.grid(row=1, column=1)
studytime_entry.grid(row=3, column=0)
fails_entry.grid(row=5, column=0)
health_entry.grid(row=5, column=1)
family_entry.grid(row=7, column=0)
free_entry.grid(row=7, column=1)
absence_entry.grid(row=3, column=1)

cog_label = tkinter.Label(user_info_frame, text="CLOSE WINDOW TO VIEW FINAL GRADE RESULT AND CORRELATION GRAPH")
cog_label.grid(row=11, column=0)

button = tkinter.Button(frame, text="Enter data", command= enter_data)
button.grid(row=12, column=0, sticky="news", padx=20, pady=10)

window.mainloop()

igrade1 = int(grade1)
igrade2 = int(grade2)
print(igrade1)
print(igrade2)
igrade1 = int(igrade1 / 5)
igrade2 = int(igrade2 / 5)
print(igrade1)
print(igrade2)

test = regression.predict([[igrade1, igrade2, studytime, failures, absences, health, family, freetime]])
print(test)

presult = test / 20 * 100

window = tkinter.Tk()
window.title("PREDICTION RESULT")
frame = tkinter.Frame(window)
frame.pack()
user_info_frame =tkinter.LabelFrame(frame, text="PREDICTION RESULT")
user_info_frame.grid(row= 0, column=0, padx=20, pady=10)
result_label = tkinter.Label(user_info_frame, text=presult)
result_label.grid(row=0, column=0)
c_label = tkinter.Label(user_info_frame, text="CLOSE WINDOW TO VIEW CORRELATION GRAPH                                  ")
c_label.grid(row=11, column=0)
window.mainloop()

print("DISPLAYING CORRELATION GRAPH")


fig, ax = plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(), annot=True,ax=ax)
plt.show()


