import torch
from torchvision import transforms
import os
import io
from PIL import Image
import logging


# Create model object
model = None

def model_handler(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.

    Example archiver:
    torch-model-archiver --model-name mnist_model --version 1.0 \
    --model-file src/models/model.py --serialized-file=model_store/deployable_model.pt \
    --handler=src/models/model_handler:model_handler

    Example torchserve:
     torchserve --start --ncs --model-store model_store \
     --models mnist_model=mnist_model.mar

    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.jit.load(model_pt_path)
    else:
        # Read bytes array as PIL image
        data = Image.open(io.BytesIO(data[0]['body']))
        # Transform PIL image to tensor
        data = transforms.ToTensor()(data)
        # Resize to 28 x 28
        data = transforms.Resize((28, 28))(data)
        #infer and return result
        pred = model(data).cpu().detach()[0].argmax().item()
        return [pred]
