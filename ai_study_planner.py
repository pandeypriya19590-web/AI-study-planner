# -------------------------------------------------
# AI STUDY PLANNER (MID-SEM MARKS FOR END-SEM PREP)
# -------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("\n-----------------------------------------")
print("     AI STUDY PLANNER — END-SEM PREP     ")
print("-----------------------------------------\n")

# Step 1: Take Inputs from User
num_subjects = int(input("Enter number of subjects: "))

subjects = []
past_hours = []
midsem_marks = []
urgency = []
days_left = []

print("\nEnter details for each subject:\n")

for i in range(num_subjects):
    sub = input(f"\nSubject {i+1} Name: ")
    h = float(input(f"  Enter your current daily study hours for {sub}: "))
    m = float(input(f"  Enter Mid-Sem Marks for {sub} (out of 50): "))
    u = int(input(f"  Enter urgency for {sub} (1-10): "))
    d = int(input(f"  Enter days left for {sub} End-Sem exam: "))

    subjects.append(sub)
    past_hours.append(h)
    midsem_marks.append(m)
    urgency.append(u)
    days_left.append(d)

# Step 2: Create DataFrame
df = pd.DataFrame({
    'Subject': subjects,
    'Past_Hours': past_hours,
    'Midsem_Marks': midsem_marks,
    'Urgency': urgency,
    'Days_Left': days_left
})

print("\nYour Input Data:")
print(df)

# Step 3: Create Target Values (Recommended Hours)
y = []
for i in range(num_subjects):
    # Lower midsem marks → need more hours
    marks_factor = max(0, (50 - midsem_marks[i]) / 10)   # punishment if marks low

    # Less days left → more hours
    days_factor = max(1, (15 - days_left[i]) * 0.3)

    # Urgency also affects hours
    urgency_factor = urgency[i] * 0.5

    rec_hours = marks_factor + days_factor + urgency_factor
    y.append(rec_hours)

y = np.array(y)

# Step 4: ML Model Training
X = df[['Past_Hours', 'Midsem_Marks', 'Urgency', 'Days_Left']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE:")
print("  Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("  R2 Score:", r2_score(y_test, y_pred))

# Step 6: Final Recommended Hours
df['Recommended_Hours'] = model.predict(X)

print("\n-----------------------------------------")
print("        FINAL END-SEM STUDY PLAN         ")
print("-----------------------------------------\n")
print(df[['Subject', 'Recommended_Hours']])

# Step 7: Visualization
plt.figure(figsize=(8, 5))
plt.bar(df['Subject'], df['Recommended_Hours'])
plt.xlabel('Subjects')
plt.ylabel('Recommended Study Hours')
plt.title('AI-Based End-Sem Study Hours Recommendation')
plt.show()
    