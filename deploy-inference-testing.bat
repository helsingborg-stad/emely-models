cp src/inference-testing/Dockerfile Dockerfile
gcloud builds submit --tag eu.gcr.io/emelybrainapi/inference-testing --timeout 3000s && gcloud run deploy --image eu.gcr.io/emelybrainapi/inference-testing --platform managed --memory 2Gi --region europe-west1 --allow-unauthenticated --timeout 900s
rm Dockerfile