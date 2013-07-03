cd Kernel
./build.sh
cd ..

cd PluginsSrc
./build.sh
cd ..

mkdir dist
cp Kernel/bin/linux/release/libkernel.so ./dist/
rm -rf dist/Plugins
mkdir dist/Plugins
cp PluginsSrc/bin/linux/release/* ./dist/Plugins/
cp Python/*.py ./dist/
