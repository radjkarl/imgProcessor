#PYPI
######
#build
python setup.py sdist #standard egg
python setup.py bdist_wheel #new_standard wheel
python setup.py install # sphinx takes only installed pkg 

#upload
python setup.py sdist upload -r pypi
python setup.py bdist_wheel upload -r pypi


#GIT
#####
git add -A #add all new files to the repo.
git commit -m "next version" #commit changes locally - set argument as message
git push origin master # Sends your commits in the "master" branch to GitHub


#API on GitHub pages
####################

sphinx-apidoc -A "Karl Bedrich" -f -M -o doc imgProcessor

cd doc/_build

git clone https://github.com/radjkarl/imgProcessor.git gh-pages

rm -r -f html
mv gh-pages html
cd html

git rm -rf -f .
git clean -fxd
cd ../../



make html
cd _build/html

#add an empty file called .nojekyll in the docs repo. This tells github’s default parsing software to ignore the sphinx-generated pages that are in the gh-pages branch
touch .nojekyll
git add .nojekyll
git commit -m "added .nojekyll"

git add .
git commit -a -m 'API updated'
git push -f origin HEAD:gh-pages 

cd ../../..
