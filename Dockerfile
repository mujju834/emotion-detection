# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the working directory
COPY . .

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install project dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "predict_emotion.py", "--server.port=8501", "--server.address=0.0.0.0"]
