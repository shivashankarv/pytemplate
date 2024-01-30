# Script that sets up the project

PROJECTNAME=$1
echo "Setting up ${PROJECTNAME}"

cd ../
pwd

sed -i '' "s@pytemplate@${PROJECTNAME}@g" environment.yml
sed -i '' "s@pytemplate@${PROJECTNAME}@g" pyproject.toml
mv pytemplate $PROJECTNAME

which conda
mamba env create -f environment.yml
eval "$(conda shell.bash hook)"
condaa activate "$PROJECTNAME"
pip install -e .

#git init
#git commit -am "Setting up the project"
#git push
