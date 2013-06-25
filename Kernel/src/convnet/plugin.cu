/* 
 * Copyright (c) 2013, Lin Min (mavenlin@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "plugin.cuh"
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>

using namespace std;

std::map<string, layerConFunc>  layers;
std::map<string, neuronConFunc> neurons;

typedef std::map<string, layerConFunc> (*layerConstructorP)  ();
typedef std::map<string, neuronConFunc> (*neuronConstructorP) ();

void loadPlugins(){
	DIR * dir;
	struct dirent * files;
	dir = opendir("Plugins");
	if(dir != NULL){
		while(files = readdir(dir)){
			if(strcmp(files->d_name, ".") == 0 || strcmp(files->d_name, "..") == 0)
				continue;
			else{
				cout<<"loading plugin "<<files->d_name<<endl;
				string plugin = "Plugins/" + string(files->d_name);
				void * handle = dlopen(plugin.c_str(), RTLD_NOW);
				if(dlerror())
					cout<<dlerror()<<endl;
				// TODO
				// Add error check code in dlsym
				// get layer constructors from current plugin
				layerConstructorP layerCon = (layerConstructorP)dlsym(handle, "layerConstructor");
				if(dlerror())
					cout<<dlerror()<<endl;
				std::map<string, layerConFunc> layerConMap = layerCon();
				layers.insert(layerConMap.begin(), layerConMap.end());
				// get neuron constructors from current plugin
				neuronConstructorP neuronCon = (neuronConstructorP)dlsym(handle, "neuronConstructor");
				std::map<string, neuronConFunc> neuronConMap = neuronCon();
				neurons.insert(neuronConMap.begin(), neuronConMap.end());
			}
		}
	}
	// This function puts all the constructors from the plugin into the global variable layers and neurons.
}








