# wildfire-smoke-detect

Object detection project to detect smoke caused by wildfire

## How to Run with Docker

1. Build docker image:

   ```bash
   docker build -t <image_name> .
   ```

   Example:

   ```bash
   docker build -t smoke-detect .
   ```

2. Run docker image:

   ```bash
   docker run -v $(pwd)/input_path:/app/input_path -v $(pwd)/output_path:/app/output_path <image_name> /app/script/inference.py /app/input_path --output /app/output_path --device cpu --weights /app/model_path

   ```

   Example:

   ```bash
   docker run -v $(pwd)/data/test/images:/app/data/test/images -v $(pwd)/output:/app/output smoke-detect python /app/script/inference.py /app/data/test/images --output /app/output --device cpu --weights /app/model/smoke-detect/weights/best.pt
   ```
