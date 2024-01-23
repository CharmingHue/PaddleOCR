mkdir /root/n
cd /root/n
apt-get update
apt-get -y install npm
npm config set registry http://registry.npmmirror.com
# npm config set registry https://registry.npm.taobao.org
# npm config set registry http://registry.npmjs.org/
npm install -g n
n 20.5.0
# n 16.20.2
# https://nodejs.org/en/about/previous-releases
hash -r
npm install -g npm@10.3.0
#npm install -g npm@8.19.4
npm install -g commitizen
npm install -g conventional-changelog-cli
cd -
commitizen init cz-conventional-changelog --save-dev --save-exact
#generate changelog
#conventional-changelog -p angular -i CHANGELOG.md -s -r 0