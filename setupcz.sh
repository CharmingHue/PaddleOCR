#!/bin/sh

if command -v sudo >/dev/null 2>&1; then
    echo "\033[32msudo is installed\033[0m"
else
    echo "\033[34minstalling sudo\033[0m"
    apt-get update
    apt-get install sudo
fi

mkdir /root/n
cd /root/n
sudo apt-get update
sudo apt-get -y install npm
sudo npm config set registry http://registry.npmmirror.com
# sudo npm config set registry https://registry.npm.taobao.org
# sudo npm config set registry https://registry.npmmirror.com/
# sudo npm config set registry http://registry.npmjs.org/
sudo npm install -g n
sudo n 20.5.0
# n 16.20.2
# https://nodejs.org/en/about/previous-releases
sudo hash -r
sudo npm install -g npm@10.3.0
#npm install -g npm@8.19.4
sudo npm install -g commitizen
sudo npm install -g conventional-changelog-cli
cd -
sudo commitizen init cz-conventional-changelog --save-dev --save-exact
#generate changelog
#conventional-changelog -p angular -i CHANGELOG.md -s -r 0