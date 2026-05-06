import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "Age": 61,
        "Gender": 1,
        "Smoking": 0,
        "Diabetes": 0,
        "Hypertension": 0,
        "Jaw_Location": 0,
        "Bone_Density": 4,
        "Bone_Height_mm": 13.7,
        "Bone_Width_mm": 4.8,
        "Implant_Length_mm": 8,
        "Implant_Diameter_mm": 4.5,
        "Immediate_Loading": 0
    }
)

print(response.json())