import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
warnings.filterwarnings("ignore")
# DATE CLEANING :
df = pd.read_csv("ML_project.csv")
df.columns = df.columns.str.strip()
drop = (["Electronic_Jamming","Latitude",'Longitude','Heading_deg','Pitch_deg','Roll_deg','Yaw_deg','Range_From_Radar_km','Bearing_deg','Doppler_Shift_Hz','Signal_to_Noise_Ratio_dB','Engine_Type','Length_m','Wingspan_m','Estimated_Mass_kg','Max_G_Load','Weather','Visibility_km','Wind_Speed_mps','Temperature_C','Humidity_pct','Sensor_Type','Detection_Time_ms','Tracking_Confidence','Track_Age_s','Number_of_Tracks','Target_Label'])
df_clean = df.drop(columns=drop,errors='ignore')
df_clean['Chaff_Flares_Detected'] = df_clean['Chaff_Flares_Detected'].map({'Yes':1,'No':0})
df_clean['Target_Class'] = pd.factorize(df_clean['Target_Class'])[0]
df_clean['Flight_Profile'] = df_clean['Flight_Profile'].map({'Ballistic':0,'Glide':1,'Combat':2,'Terrain Following':3,'Hover': 4,'Cruise': 5,'Loiter': 6})
df_clean['Infrared_Intensity'] = df_clean['Infrared_Intensity'].map({'Low':0,'Medium':1,'High':2,'Very High':3})
df_clean['Heat_Signature'] = df_clean['Heat_Signature'].map({'Low':0,'Medium':1,'High':2,'Very High':3})
df_clean['Stealth_Category'] = df_clean['Stealth_Category'].map({'Low':0,'Medium':1,'High':2})
df_clean['Threat_Level'] = df_clean['Threat_Level'].map({'Low':0,'Medium':1,'High':2})
df_clean['Radar_Band'] = df_clean['Radar_Band'].map({'C':0,'X':1,'S':2,'L':3})
#FEATURE ENGINEERING AND EXTRACTION :
y = df_clean['Target_Class']
scaler = StandardScaler()
#np.random.seed(42)
x = df_clean.drop(columns = ['Target_Class','Object_ID','Flight_Profile','Threat_Level'],errors='ignore')
#noise_mask = np.random.rand(len(df_clean))<0.01
#df_clean.loc[noise_mask,'Target_Class'] = np.random.choice(df_clean['Target_Class'].unique(),size=noise_mask.sum())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42,stratify=y)
#x = df_clean.drop(columns=['Target_Class','Object_ID'])
cols = ['Speed_mps','Altitude_m','Climb_Rate_mps','Acceleration_mps2','Turn_Rate_degps','Radar_Cross_Section_m2','Radar_Return_Strength','Chaff_Flares_Detected','Infrared_Intensity','Heat_Signature','Stealth_Category','Radar_Band']
x_train[cols] = scaler.fit_transform(x_train[cols])
x_test[cols] = scaler.transform(x_test[cols])
model = XGBClassifier(max_depth=2,random_state=42,learning_rate=0.01,colsample_bytree=0.4,subsample=0.5)
model.fit(x_train,y_train)
importance = pd.Series(model.feature_importances_,index=x_train.columns)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
print("X columns : ",x.columns)
print(importance.sort_values(ascending=False))
scores = cross_val_score(model,x_train,y_train,cv =5)
import joblib

# Model save karo
joblib.dump(model, 'air_defence_detection.pkl')
print("Model saved successfully!")