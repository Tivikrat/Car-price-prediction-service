docker stop $(cat docker.cid)
docker rm $(cat docker.cid)
rm -f docker.cid
