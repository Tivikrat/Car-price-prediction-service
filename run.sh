docker build -t='predictor:v1' .
docker run --cidfile docker.cid -d -p 5000:5000 --name predictor predictor:v1
docker port predictor 5000 > docker.url
url=$(cat docker.url)
xdg-open "${url/0.0.0.0/http://127.0.0.1}"
