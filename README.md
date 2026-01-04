Instructions to execute the project 

1) Open Github respository link shared in the project
   
2) Download the project to your device
   
3) Open terminal / command prompt: python src/train.py
   This script trains Logistic Regression and Random Forest models using a unified preprocessing pipeline and saves the best model for deployment
   
4) Command: mlflow ui Open browser http://127.0.0.1:5000
   MLflow is used to track experiments, metrics, and artifacts, enabling reproducibility and comparison across models
   
5) Command uvicorn api.app:app --reload
Open browser: http://127.0.0.1:8000/docs
The trained model is exposed as a REST API using FastAPI, providing predictions and confidence scores.

6) Docker build command: docker build -t heart-disease-api .
   Run container: docker run -p 8000:8000 heart-disease-api
   Open: http://127.0.0.1:8000/docs
The FastAPI service is containerized using Docker to ensure portability and reproducibility.

7) Deploy :kubectl apply -f k8s/deployment.yaml
           kubectl apply -f k8s/service.yaml
   Verify: kubectl get pods
           kubectl get services
   Open: http://localhost:30007/docs
The Dockerized application is deployed to a local Kubernetes cluster using Docker Desktop and exposed via a NodePort service.

8) Go to GitHub → Actions, Open the latest green workflow
   Click: build-and-test
      •	steps will be shown:
      o	Install dependencies
      o	Linting
      o	Unit tests
      o	Training step (skipped in CI)
A GitHub Actions CI/CD pipeline automates linting, testing, and training validation on every push, ensuring code quality and reproducibility

9) After calling /predict, see the terminal logs:
    POST /predict Status=200 Time=...
    GET /health Status=200 Time=...
  Open: http://127.0.0.1:8000/health
Basic monitoring is implemented using structured logging and a health check endpoint to track API availability and performance

10)This project demonstrates a complete MLOps workflow, from model development to production deployment, incorporating CI/CD, containerization, orchestration, and monitoring using industry-standard tools.



  
   
