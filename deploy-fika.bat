cp src/fika/Dockerfile Dockerfile
gcloud builds submit --tag eu.gcr.io/emelybrainapi/blender-90m --timeout 3000s && gcloud run deploy --image eu.gcr.io/emelybrainapi/blender-90m --platform managed --memory 4Gi --cpu 4 --region europe-west3 --allow-unauthenticated --timeout 900s
rm Dockerfile