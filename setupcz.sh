mkdir /root/n
cd /root/n
apt-get update
apt-get -y install npm
npm config set registry https://registry.npm.taobao.org
# npm config set registry http://registry.npmmirror.com
npm install -g n
n 20.5.0
hash -r
npm install -g npm@10.3.0
npm install -g commitizen
npm install -g conventional-changelog-cli
cd -
commitizen init cz-conventional-changelog --save-dev --save-exact