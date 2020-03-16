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

#仮想環境下　
#新しいアプリの開発　https://docs.djangoproject.com/ja/3.0/intro/tutorial01/ここを見ればわかる
 django-admin.exe startproject hoge . #新しくプロジェクトを作る宣言
 python manage.py startapp blog　#あたらしくアプリを作成すると宣言 mysiteのなか

 git remote rm origin # https://qiita.com/yu-ki0718/items/3c8aae2c81ca3f82f522

#commandline github
$ git status
[...]
$ git add --all .
$ git status
[...]
$ git commit -m "Modified templates to display posts from database."

#python_anywhere
cd ~/naoyayg.pythonanywhere.com
workon naoyayg.pythonanywhere.com
git pull
[...]
$ git push


#以下自作メモ
#ディレクトリ名 test
# 仮想環境作成　python -m venv machidayg  仮想環境動作　machidayg\Scripts\activate
#仮想環境上でdjango install python -m pip install --upgrade pip
#ディレクトリの中に requirements.txtを入れる。 +  pip install -r requirements.txt
#プロジェクト作成 django-admin.exe startproject mathmath.
#設定変更　mathmathnのなかのsettings 言語とか時間
#python manage.py migrateして、動けばOK

#アプリ作成を始める。
# python manage.py startapp math1
#settingsをみて、INSTALLED_APPSに'math1.apps.Math1Config'を追加
#python manage.py makemigrations math1と打って、math1というappを作ったのを知らせる。
#viewで/adminではないほうのviewを決める。+math1上にurl.pyを作る。
#url.py(math1)で、urlの登録 path("",views.yourname(defの名前),name = "yourname"(名前付け)) 小ページに行くイメージ (ブログで言うとこの草稿とか)
#mathmathのurl.pyにurlに登録する    path("yourname/(飛ぶページのurlを指定)",include("math1.urls")) 元のページ（/admin等々の切り替え）
#
