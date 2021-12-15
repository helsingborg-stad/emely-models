# Usage ex:
# sh deploy.sh -m blender_90M -s fika 
# sh deploy.sh -m jani-peralez -s interview

while getopts m:s: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        s) service=${OPTARG};;
    esac
done
echo "Deploying to GCP!";
echo "model: $model";
echo "service: $service";

# Generate the gcloudignore file
python src/utils/generate_gcloudignore.py --model $model;echo "Generated a .gcloudignore... ";

# Something to check if the python script crashed

# Deploy the service
if [ $service = "fika" ];
then
    echo "deploying to fika...";
    gcloud builds submit --config=cloudbuild.yaml --substitutions=_MODEL=$model,_SERVICE_NAME="blender-90m";
elif [ $service = "interview" ];
then
    echo "deploying to interview...";
    gcloud builds submit --config=cloudbuild.yaml --substitutions=_MODEL=$model,_SERVICE_NAME="interview-model";
else
  echo "INVALID SERVICE ARGUMENT: fika or interview allowed";
fi

rm .gcloudignore; echo "Removing the generated .gcloudignore"