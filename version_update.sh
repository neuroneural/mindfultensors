version=`cat version.txt`

sed -i -e "1s/s = .*/s = $version/" version_update.py
rm *.py-e || true

updated_version=$(python version_update.py)
echo $updated_version > version.txt

git add version.txt version_update.py
git commit -m "version update to $updated_version"
git push origin version
git checkout master

sed -i -e "1s/__version_info__ = .*/__version_info__ = $updated_version/" version.py
rm *.py-e || true
