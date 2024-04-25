import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import aiplatform

# Path to your service account key file
service_account_key = '/path/to/your/service_account_key.json'

# Path to your data directory
data_directory = '/home/kira/Desktop/SunHack/Training Dataset/Data/'

# Initialize Google Cloud Storage client
credentials = service_account.Credentials.from_service_account_file(service_account_key)
gcs_client = storage.Client(credentials=credentials)

# Initialize AI Platform client
aiplatform.init(project='your-project-id', credentials=credentials)

# Read the data from multiple CSV files
data_frames = []
for i in range(1, 60):  # Assuming you have 59 CSV files
    file_name = f"DCIOT_DATA_202402{i:02d}_0001 TO 202402{i+1:02d}_0001.csv"
    file_path = os.path.join(data_directory, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data_frames.append(df)

# Concatenate all data frames into a single DataFrame
all_data = pd.concat(data_frames)

# Split the data into features (X) and target (y)
X = all_data[['humidity1', 'humidity2', 'humidity3', 'humidity4', 'humidity5', 'humidity6', 'humidity7', 'humidity8',
              'temperature1', 'temperature2', 'temperature3', 'temperature4', 'temperature5', 'temperature6', 'temperature7', 'temperature8']]
y = all_data['temperature8']  # Assuming we are predicting temperature8

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Upload trained model to AI Platform
model_path = 'gs://your-bucket-name/model'  # Replace with your GCS bucket name and desired model path
model_full_path = aiplatform.Model.upload(
    model_display_name='my_model',
    artifact_uri=model_path,
    serving_container_image_uri=aiplatform.gapic.ModelServiceClient.DEFAULT_CONTAINER_VERSION_GPU,
)

# Deploy the model
endpoint = model_full_path.deploy(
    machine_type='n1-standard-4',  # Customize machine type as needed
    accelerator_type='NVIDIA_TESLA_T4',  # Customize accelerator type as needed
    accelerator_count=1,
)
