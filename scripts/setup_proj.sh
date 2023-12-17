# Script that sets up the project

PROJECTNAME=$1

sed -i 's/pytemplate/${PROJECTNAME}/g' environment.yml
sed -i 's/pytemplate/${PROJECTNAME}/g' pyproject.toml
mv pytemplate $PROJECTNAME

git init
git commit -am "Setting up the project"

conda env create -f environment.yml
