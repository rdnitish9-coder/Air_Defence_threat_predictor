# AIR DEFENCE THREAT PREDICTOR 
An end-to-end Machine Learning web application deployed on Streamlit to classify incoming threat via radar and physical inputs  
**[https://airdefencethreatpredictor-89vvzvswchz7bpudhfa4m4.streamlit.app]**
--
## FEATURES : 
-**Real Time Prediction :** Gives highly accurate prediction , when connected to a real time radar , uses the input features as :   
-Speed  
-Altitude  
-Acceleration  
-Turn_rate_degps  
-Range_from_radar_km   ....and 29 other features  
-**Currently uses a demo generated data, can be industrialized by using much more accurate data.  
UI built with Streamlit.    
## Tech Stack : 
* **Language used :** Python
* **Machine learning :** Scikit-Learn ,Pandas, Numpy .XGboost.
* **Frontend & Deployment :** Streamlit
## To run Locally : 

# 1.Clone the repository
git clone [https://github.com/rdnitish9-coder/Air_Defence_threat_predictor.git]
# 2.Install required dependencies
pip install -r requirements.txt
# 3.Launch the streamlit app
streamlit run app.py








