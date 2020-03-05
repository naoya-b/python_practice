#memo https://tutorial.djangogirls.org/ja/installation/#pythonanywhere
#create new directori
mkdir django-girls
cd django-girls

#仮想環境作成 django girls上
python -m venv naoyg
naoyg\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


#2回目以降
cd django-girls
naoyg\Scripts\activate #仮想環境

#commandline上
~/djangogirls$
python manage.py migrate
python manage.py runserver #djangoserverが動く

#新しいアプリの開発
 python manage.py startapp blog

 git remote rm origin # https://qiita.com/yu-ki0718/items/3c8aae2c81ca3f82f522

#commandline github
$ git status
[...]
$ git add --all .
$ git status
[...]
$ git commit -m "Modified templates to display posts from database."

#python_anywhere
git pull
[...]
$ git push
