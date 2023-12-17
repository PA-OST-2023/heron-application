sudo apt install git
sudo apt-add-repository ppa:deadsnakes/ppa -y
sudo apt install python3.8
sudo apt install python3.8-venv
setup ssh key for git
ssh-keygen -o
cat /<keyname>.pub

git clone git@github.com:PA-OST-2023/heron-application.git
cd heron-application
python3.8 -m venv venv
source venv/bin/activate
pyaudio:
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg 
sudo apt install python3.8-dev
sudo apt install python3-pyaudio
