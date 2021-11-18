cp src/interview/Dockerfile Dockerfile
gcloud builds submit --tag eu.gcr.io/emely-gcp/interview-model --timeout 3000s && gcloud run deploy interview-model --image eu.gcr.io/emely-gcp/interview-model --platform managed --memory 4Gi --region europe-west3 --allow-unauthenticated --timeout 900s
rm Dockerfile