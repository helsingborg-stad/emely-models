from pathlib import Path
from argparse import ArgumentParser

model_dir = Path(__file__).resolve().parents[2] / "models"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model dir to ignore")

    args = parser.parse_args()

    models_to_ignore = []
    model_found = False
    for item in model_dir.iterdir():
        if item.is_dir():
            if item.name == args.model:
                model_found = True
            else:
                models_to_ignore.append("models/" + item.name)

    if not model_found:
        raise Exception(
            "Did not find the specified model! This could disrupt the deployment"
        )

    with open("gcloudignore-standard.txt", "r") as f:
        text = f.read()

    gcloudignore = text + "\n" + "\n".join(models_to_ignore)

    with open(".gcloudignore", "w") as f:
        f.write(gcloudignore)
