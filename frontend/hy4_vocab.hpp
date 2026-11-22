#pragma once

#include "llama.h"
#include <string>

llama_model * load_hy4_vocab(const std::string & path, llama_model_params params, std::string & err);
