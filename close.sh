sudo docker stop $(cat docker.cid)
sudo docker rm $(cat docker.cid)
rm -f docker.cid
