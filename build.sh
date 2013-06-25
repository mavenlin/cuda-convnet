cd Kernel
./build.sh
cd ..

cd PluginsSrc
./build.sh
cd ..

cp Kernel/bin/linux/release/libkernel.so ./
rm -rf Plugins
mkdir Plugins
cp PluginsSrc/bin/linux/release/* ./Plugins/

