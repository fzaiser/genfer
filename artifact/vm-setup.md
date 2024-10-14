## Commands to set up the artifact VM

Username: user

Password: 123

## Dependencies

```shell
sudo apt install curl git build-essential

mkdir artifact
ln -s ~/artifact ~/Desktop/artifact
cd artifact/
git clone https://github.com/fzaiser/genfer.git
mv genfer tool
cd tool
git checkout loops
cd ..
ln -s tool/README.md README.md

# set up our tool:
cd tool
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install libclang-dev z3 libz3-dev coinor-cbc coinor-libcbc-dev coinor-libipopt-dev
cargo build --release --bins
cd ..

# set up evaluation of our tool:
sudo apt install python-is-python3 python3-pip
pip install numpy matplotlib

# set up Polar:
sudo apt install python3.10-venv # for Polar
git clone https://github.com/fzaiser/polar.git
cd polar
pip install --user virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
exit
cd ..

# set up GuBPI:
sudo apt install dotnet-sdk-7.0 # for GuBPI
git clone https://github.com/gubpi-tool/gubpi.git
cd gubpi
cd vinci
make
cd ..
cd src
dotnet build -c "Release" -o ../app
cd ..
cp vinci/vinci ./app
cd ..
```

## Other stuff

* adjust power settings (don't go sleep on inactivity)
* add English keyboard layout (in addition to German (Neo2))
* list view instead of icons in file explorer
