python -m venv env
. env/Scripts/activate
python -m pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python pillow transformers
