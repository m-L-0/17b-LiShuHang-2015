from flask import Flask, render_template, redirect, url_for, session
from flask_wtf.file import FileField, FileAllowed
from werkzeug.utils import secure_filename
from wtforms import SubmitField
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
import os
from cnn import cnn
from multiprocessing import Pool

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(36)
bootstrap = Bootstrap(app)


class Form(FlaskForm):
    File = FileField(
        '请上传验证码：',
        validators=[
            FileAllowed(['png', 'jpg', 'jpeg', 'gif'])
        ])
    submit = SubmitField('ｕｐｌｏａｄ')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form()
    if form.validate_on_submit():
        file = form.File.data
        filename = secure_filename(file.filename)
        img_path = basedir + '/' + filename
        file.save(img_path)
        pool = Pool()
        res = pool.apply_async(cnn, (img_path, ))

        session['result'] = res.get()
        os.remove(img_path)
        return redirect(url_for('index'))
    return render_template(
        'index.html', form=form, result=session.get('result'))


if __name__ == '__main__':
    app.run()
