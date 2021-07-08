cp src/interview/Dockerfile Dockerfile
gcloud builds submit --tag eu.gcr.io/emelybrainapi/interview-model --timeout 3000s && gcloud run deploy --image eu.gcr.io/emelybrainapi/interview-model --platform managed --memory 4Gi --cpu 4 --region europe-west3 --allow-unauthenticated --timeout 900s
rm Dockerfile