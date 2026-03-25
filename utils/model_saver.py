import torch
import os


def save_model(model, dataset_name, stage, model_name):
    base_dir = os.path.join(
        "models",
        "trained_models",
        dataset_name,
        stage
    )

    # Create directories if they do not exist
    os.makedirs(base_dir, exist_ok=True)

    model_path = os.path.join(base_dir, model_name)

    torch.save(model.state_dict(), model_path)

    print("Model saved at:", model_path)



def load_model(model, dataset_name, stage, model_name, device):
    model_path = os.path.join(
        "models",
        "trained_models",
        dataset_name,
        stage,
        model_name
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    print("Model loaded from:", model_path)

    return model