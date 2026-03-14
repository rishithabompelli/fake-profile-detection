from flask import Flask, render_template, request
from predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    followers = int(request.form.get('followers', 0))
    following = int(request.form.get('following', 0))
    follower_following_ratio = float(request.form.get('follower_following_ratio', 0))
    posts = int(request.form.get('posts', 0))
    has_profile_pic = int(request.form.get('has_profile_pic', 0))
    username_randomness = int(request.form.get('username_randomness', 0))
    suspicious_links_in_bio = int(request.form.get('suspicious_links_in_bio', 0))
    verified = int(request.form.get('verified', 0))
    bio_length = int(request.form.get('bio_length', 0))

    result = predict(followers, following, follower_following_ratio,
                     posts, has_profile_pic, username_randomness,
                     suspicious_links_in_bio, verified, bio_length)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=False)